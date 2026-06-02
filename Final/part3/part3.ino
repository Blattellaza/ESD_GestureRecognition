// ============================================================
// ESD Gesture Recognition — FreeRTOS Version for Arduino Due
// Include: <FreeRTOS.h> <task.h> <semphr.h>
// Target : Arduino Due (SAM3X8E, 96KB SRAM, 84MHz)
//
// 改動摘要（相對於 part3__1_.ino）：
//   - 新增 FreeRTOS 三任務架構（Button / Sample / Inference）
//   - xTaskCreate 使用 FreeRTOS heap（Due 96KB SRAM 充裕）
//   - 所有業務資料（nn_input/buffer_A/ts_us/MAFilter）全部 static
//   - TaskSample 使用 vTaskDelayUntil 確保固定週期，消除 jitter
//   - Button 邏輯、MAFilter、Static Filter、NN pipeline 完全不變
//   - [SMOOTH] 輸出格式與原版完全相容，analyze_log.py 可直接解析
// ============================================================

#include "I2C_GPIO.h"
#include "nn_ops.h"
#include <string.h>
#include <math.h>

// Arduino Due 專用 FreeRTOS headers
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>

#define I2C_use_GPIO 1

static const uint8_t MPU_ADDR = 0x68;
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

#define BTN_PIN           12
#define DEBOUNCE_MS       30

#define SAMPLE_RATE_HZ        100
#define SAMPLE_INTERVAL_MS    10        // 100 Hz = 10 ms per sample
#define RECORD_SECONDS        1.5f
#define NUM_SAMPLES           150       // 100 Hz × 1.5 s

// ============================================================
// Moving Average Filter（與原版完全相同）
// ============================================================
#define WINDOW_SIZE 5

typedef struct {
  int16_t buf[WINDOW_SIZE];
  int     head;
  int     count;
  int32_t sum;
} MAFilter;

static MAFilter ma_ax;
static MAFilter ma_ay;
static MAFilter ma_az;

static int16_t ma_update(MAFilter *f, int16_t val) {
  if (f->count == WINDOW_SIZE) {
    f->sum -= (int32_t)f->buf[f->head];
  } else {
    f->count++;
  }
  f->buf[f->head] = val;
  f->sum += (int32_t)val;
  f->head = (f->head + 1) % WINDOW_SIZE;
  return (int16_t)(f->sum / f->count);
}

// ============================================================
// Static Filter（與原版完全相同）
// ============================================================
#define STATIC_ACCEL_THRESH  0.15f
#define STATIC_GYRO_THRESH   25.0f
#define STATIC_FRAMES        10
#define STATIC_RATIO_THRESH  0.60f

static float prev_accel_mag = 1.0f;
static int   static_counter = 0;

// ============================================================
// SRAM 緩衝區（全部 static，與原版相同）
// ============================================================
static int8_t nn_input[NUM_SAMPLES][6];     //  900 bytes
#define LAYER_BUFFER_SIZE 2400
static int8_t buffer_A[LAYER_BUFFER_SIZE];  // 2400 bytes
static int8_t logits[NN_NUM_CLASSES];       //    3 bytes

// Jitter 統計時間戳（static，150 × 4 = 600 bytes）
static uint32_t ts_us[NUM_SAMPLES];

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

// ============================================================
// 任務間共享狀態（volatile 確保跨任務可見性）
// ============================================================
static volatile bool     isRecording        = false;
static volatile int      static_frame_count = 0;
static volatile uint32_t t_first_sample_us  = 0;
static volatile uint32_t t_last_sample_us   = 0;
static volatile uint32_t sampleId          = 0;

// ============================================================
// Button debounce 狀態（與原版完全相同）
// ============================================================
static unsigned long lastDebounceMs    = 0;
static int           lastButtonReading = HIGH;
static int           buttonState       = HIGH;

// ============================================================
// FreeRTOS Semaphore handles
// ============================================================
static SemaphoreHandle_t xStartSem = NULL;  // Button → TaskSample
static SemaphoreHandle_t xInferSem = NULL;  // TaskSample → TaskInference

// ============================================================
// 函式宣告
// ============================================================
void    MPU6050_wakeup();
bool    readImu6_raw(int16_t*, int16_t*, int16_t*,
                     int16_t*, int16_t*, int16_t*);
bool    isStatic(int16_t, int16_t, int16_t,
                 int16_t, int16_t, int16_t);
int8_t  quantize_single_axis(int16_t, int32_t, int32_t, uint8_t);
void    runInference(int, uint32_t, uint32_t);

static TaskHandle_t hButton = NULL;  // Task handle，供 getFreeRam() 查詢
static TaskHandle_t hSample = NULL;
static TaskHandle_t hInfer  = NULL;

static void getFreeRam() {
  // uxTaskGetStackHighWaterMark() 回傳各 Task stack 歷史最低剩餘 words
  // 加總後 * 4 換算為 bytes（ARM Cortex-M3，1 word = 4 bytes）
  // 原版 sbrk(0) 在 FreeRTOS 下因 heap 指標被推高而回傳負數，改用此方式
  UBaseType_t total = uxTaskGetStackHighWaterMark(hButton)
                    + uxTaskGetStackHighWaterMark(hSample)
                    + uxTaskGetStackHighWaterMark(hInfer);

  Serial.print("Free SRAM: ");
  Serial.print(total * 4);
  Serial.println(" bytes");
}


// ============================================================
// MPU6050_wakeup()（與原版相同）
// ============================================================
void MPU6050_wakeup() {
  I2C_start();
  if (!I2C_write_byte((MPU_ADDR << 1) | 0x00) ||
      !I2C_write_byte(PWR_MGMT_1)              ||
      !I2C_write_byte(0x00)) {
    Serial.println("# ERROR: Failed to wake up MPU6050");
  }
  I2C_stop();
}

// ============================================================
// readImu6_raw()（與原版相同）
// ============================================================
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz) {
  uint8_t buf[14];
  I2C_start();
  if (!I2C_write_byte(MPU_ADDR << 1) ||
      !I2C_write_byte(ACCEL_XOUT_H)) {
    I2C_stop(); return false;
  }
  I2C_repeated_start();
  if (!I2C_write_byte((MPU_ADDR << 1) | 1)) {
    I2C_stop(); return false;
  }
  for (uint8_t i = 0; i < 13; i++) buf[i] = I2C_read_byte(true);
  buf[13] = I2C_read_byte(false);
  I2C_stop();

  *ax = (int16_t)((buf[0]  << 8) | buf[1]);
  *ay = (int16_t)((buf[2]  << 8) | buf[3]);
  *az = (int16_t)((buf[4]  << 8) | buf[5]);
  *gx = (int16_t)((buf[8]  << 8) | buf[9]);
  *gy = (int16_t)((buf[10] << 8) | buf[11]);
  *gz = (int16_t)((buf[12] << 8) | buf[13]);
  return true;
}

// ============================================================
// quantize_single_axis()（與原版相同）
// ============================================================
int8_t quantize_single_axis(int16_t raw_val,
                             int32_t s, int32_t z, uint8_t shift) {
  int32_t q = ((int32_t)raw_val * s + z) >> shift;
  if (q >  127) q =  127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// ============================================================
// isStatic()（與原版相同）
// ============================================================
bool isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw) {
  float ax_g = ax_raw / 16384.0f;
  float ay_g = ay_raw / 16384.0f;
  float az_g = az_raw / 16384.0f;
  float gx_d = gx_raw / 131.0f;
  float gy_d = gy_raw / 131.0f;
  float gz_d = gz_raw / 131.0f;

  float accel_mag = sqrtf(ax_g*ax_g + ay_g*ay_g + az_g*az_g);
  float delta_mag = fabsf(accel_mag - prev_accel_mag);
  prev_accel_mag  = accel_mag;
  float gyro_mag  = sqrtf(gx_d*gx_d + gy_d*gy_d + gz_d*gz_d);

  if (delta_mag < STATIC_ACCEL_THRESH && gyro_mag < STATIC_GYRO_THRESH) {
    static_counter++;
  } else {
    static_counter = 0;
  }
  return (static_counter >= STATIC_FRAMES);
}

// ============================================================
// runInference()（與原版相同）
// ============================================================
void runInference(int sf_count, uint32_t t_first, uint32_t t_last) {
  float static_ratio = (float)sf_count / (float)NUM_SAMPLES;

  if (static_ratio >= STATIC_RATIO_THRESH) {
    uint32_t t_done = micros();
    Serial.println("# INFERENCE_RESULT: static");
    Serial.print("# static_ratio: "); Serial.println(static_ratio);
    Serial.print("[RESULT] trial=");  Serial.print(sampleId);
    Serial.print(" first=");          Serial.print(t_first);
    Serial.print(" last=");           Serial.print(t_last);
    Serial.print(" pred=");           Serial.println(t_done);
    return;
  }

  uint32_t startUs = micros();

  // Step 1. Conv1 + ReLU
  conv1d(&nn_input[0][0], buffer_A,
         CONV1_W, CONV1_B,
         NUM_SAMPLES, 6, NN_CONV1_OUT_CH,
         RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu(buffer_A, NUM_SAMPLES * NN_CONV1_OUT_CH);

  // Step 2. MaxPool1
  maxpool1d((int8_t(*)[NN_CONV1_OUT_CH])buffer_A,
            (int8_t(*)[NN_CONV1_OUT_CH])(&buffer_A[1200]),
            NUM_SAMPLES);

  // Step 3. Conv2 + ReLU
  conv1d(&buffer_A[1200], buffer_A,
         CONV2_W, CONV2_B,
         NUM_SAMPLES / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu(buffer_A, (NUM_SAMPLES / 2) * NN_CONV2_OUT_CH);

  // Step 4. 後端借用
  int8_t* gap_out    = &buffer_A[1200];
  int8_t* dense1_out = &buffer_A[1232];

  // Step 5. GAP
  global_avg_pool((int8_t*)buffer_A, gap_out,
                  NUM_SAMPLES / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // Step 6. Dense1 + ReLU
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);

  // Step 7. Dense2 (logits)
  dense(dense1_out, logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  uint32_t latencyUs = micros() - startUs;

  int label = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[label]) label = i;
  }

  uint32_t t_done = micros();

  Serial.print("# INFERENCE_LATENCY_US,"); Serial.println(latencyUs);
  Serial.print("# PREDICTION,");           Serial.println(LABEL_NAMES[label]);
  Serial.print("# static_ratio: ");        Serial.println(static_ratio);
  Serial.print("[RESULT] trial=");  Serial.print(sampleId);
  Serial.print(" first=");          Serial.print(t_first);
  Serial.print(" last=");           Serial.print(t_last);
  Serial.print(" pred=");           Serial.println(t_done);
}

// ============================================================
// TaskButton — 優先級 1
// 與原版 handleButton() 邏輯完全相同，不更動任何 debounce 機制
// 偵測到按下後 xSemaphoreGive(xStartSem) 通知 TaskSample
// ============================================================
static void TaskButton(void *pvParam) {
  (void)pvParam;

  for (;;) {
    // ---- 與原版 handleButton() 完全相同的 debounce 邏輯 ----
    int reading = digitalRead(BTN_PIN);

    if (reading != lastButtonReading) {
      lastDebounceMs = millis();
    }

    if ((millis() - lastDebounceMs) > DEBOUNCE_MS) {
      if (reading != buttonState) {
        buttonState = reading;
        // 按下（LOW）且目前不在錄製中 → 通知 TaskSample 啟動
        if (buttonState == LOW && !isRecording) {
          xSemaphoreGive(xStartSem);
        }
      }
    }

    lastButtonReading = reading;
    // ---------------------------------------------------------

    vTaskDelay(pdMS_TO_TICKS(5));  // 5ms 輪詢，足夠 debounce 精度
  }
}

// ============================================================
// TaskSample — 優先級 3（最高）
//
// 核心改動：vTaskDelayUntil 取代原版 busy-wait
//   原版：while(micros() < nextSampleUs) {}
//         → Serial.print 佔用 CPU，造成每個週期長度不一致
//   RTOS：vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(10))
//         → 以絕對 tick 喚醒，即使本次迭代延遲，
//           下次喚醒點仍固定，誤差不累積
//
// [SMOOTH] jitter 統計格式與原版完全相容
// ============================================================
static void TaskSample(void *pvParam) {
  (void)pvParam;

  for (;;) {
    // 等待 TaskButton 發出的開始信號
    xSemaphoreTake(xStartSem, portMAX_DELAY);

    isRecording = true;

    // ---- 重置所有狀態（與原版 recordOneSample() 開頭相同）----
    static_counter     = 0;
    prev_accel_mag     = 1.0f;
    static_frame_count = 0;
    memset(&ma_ax, 0, sizeof(MAFilter));
    memset(&ma_ay, 0, sizeof(MAFilter));
    memset(&ma_az, 0, sizeof(MAFilter));
    // -----------------------------------------------------------

    Serial.println("# GET_READY");
    Serial.print("# WINDOW_SIZE="); Serial.println(WINDOW_SIZE);
    vTaskDelay(pdMS_TO_TICKS(1000));  // 1 秒準備時間

    Serial.println("# START");
    Serial.println("sample_id,timestep,t_us,"
                   "ax_raw,ay_raw,az_raw,"
                   "ax_filt,ay_filt,az_filt,"
                   "gx,gy,gz");

    // ---- vTaskDelayUntil 精確週期取樣 ----
    // xLastWakeTime 記錄「上次被喚醒的 tick」
    // 每次呼叫後自動推進 SAMPLE_INTERVAL_MS ticks
    // 即使 I2C 讀取或 Serial 輸出有微小延遲，
    // 下一個喚醒點仍然是絕對固定的時間槽
    TickType_t xLastWakeTime = xTaskGetTickCount();

    for (int i = 0; i < NUM_SAMPLES; i++) {

      // 精確等待到下一個 10ms 時間槽（核心 jitter 消除機制）
      vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));

      // 記錄取樣時間戳（用於 jitter 統計）
      uint32_t tUs = micros();
      ts_us[i] = tUs;
      if (i == 0)             t_first_sample_us = tUs;
      if (i == NUM_SAMPLES-1) t_last_sample_us  = tUs;

      int16_t ax, ay, az, gx, gy, gz;
      bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

      if (!ok) {
        Serial.print("# ERROR at timestep "); Serial.println(i);
        // 標記錯誤：static_frame_count = NUM_SAMPLES 讓推論判定為 static 跳過
        static_frame_count = NUM_SAMPLES;
        xSemaphoreGive(xInferSem);
        isRecording = false;
        goto next_trial;
      }

      // Moving Average Filter（與原版完全相同）
      int16_t ax_f = ma_update(&ma_ax, ax);
      int16_t ay_f = ma_update(&ma_ay, ay);
      int16_t az_f = ma_update(&ma_az, az);

      // 量化存入 nn_input（與原版完全相同）
      nn_input[i][0] = quantize_single_axis(ax_f, PRE_S[0], PRE_Z[0], PRE_SHIFT[0]);
      nn_input[i][1] = quantize_single_axis(ay_f, PRE_S[1], PRE_Z[1], PRE_SHIFT[1]);
      nn_input[i][2] = quantize_single_axis(az_f, PRE_S[2], PRE_Z[2], PRE_SHIFT[2]);
      nn_input[i][3] = quantize_single_axis(gx,   PRE_S[3], PRE_Z[3], PRE_SHIFT[3]);
      nn_input[i][4] = quantize_single_axis(gy,   PRE_S[4], PRE_Z[4], PRE_SHIFT[4]);
      nn_input[i][5] = quantize_single_axis(gz,   PRE_S[5], PRE_Z[5], PRE_SHIFT[5]);

      // isStatic 使用濾波後加速度計值（與原版相同）
      if (isStatic(ax_f, ay_f, az_f, gx, gy, gz)) {
        static_frame_count++;
      }

      // CSV 輸出（與原版完全相同）
      Serial.print(sampleId); Serial.print(',');
      Serial.print(i);        Serial.print(',');
      Serial.print(tUs);      Serial.print(',');
      Serial.print(ax);       Serial.print(',');
      Serial.print(ay);       Serial.print(',');
      Serial.print(az);       Serial.print(',');
      Serial.print(ax_f);     Serial.print(',');
      Serial.print(ay_f);     Serial.print(',');
      Serial.print(az_f);     Serial.print(',');
      Serial.print(gx);       Serial.print(',');
      Serial.print(gy);       Serial.print(',');
      Serial.println(gz);
    }

    // ---- Jitter 統計（[SMOOTH] 格式與原版完全相同）----
    {
      int   n     = NUM_SAMPLES - 1;   // 149 個間隔
      float sum_p = 0.0f;
      for (int i = 1; i < NUM_SAMPLES; i++) {
        sum_p += (float)(ts_us[i] - ts_us[i-1]);
      }
      float mean_p = sum_p / (float)n;

      // 兩步計算法，避免大數相減精度損失
      float var_sum = 0.0f;
      for (int i = 1; i < NUM_SAMPLES; i++) {
        float dev = (float)(ts_us[i] - ts_us[i-1]) - mean_p;
        var_sum += dev * dev;
      }
      float std_p = sqrtf(var_sum / (float)n);

      float max_dev = 0.0f;
      for (int i = 1; i < NUM_SAMPLES; i++) {
        float dev = fabsf((float)(ts_us[i] - ts_us[i-1])
                          - (float)(SAMPLE_INTERVAL_MS * 1000UL));
        if (dev > max_dev) max_dev = dev;
      }

      // [SMOOTH] 格式與原版相同，analyze_log.py 可直接解析
      Serial.print("[SMOOTH] window=");  Serial.print(WINDOW_SIZE);
      Serial.print(" mean=");            Serial.print(mean_p, 1);
      Serial.print(" std=");             Serial.print(std_p, 1);
      Serial.print(" max_dev=");         Serial.println(max_dev, 1);
    }
    // ---------------------------------------------------

    // 通知 TaskInference 開始推論
    xSemaphoreGive(xInferSem);

    next_trial:;  // I2C 錯誤時跳至此處，回到等待狀態
  }
}

// ============================================================
// TaskInference — 優先級 2
// 等待 xInferSem → 執行 runInference() → 更新 sampleId
// ============================================================
static void TaskInference(void *pvParam) {
  (void)pvParam;

  for (;;) {
    xSemaphoreTake(xInferSem, portMAX_DELAY);

    runInference((int)static_frame_count,
                 t_first_sample_us,
                 t_last_sample_us);

    Serial.println("# END");
    getFreeRam();
    sampleId++;
    isRecording = false;

    Serial.println("Push btn to start recording.");
  }
}

// ============================================================
// setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(BTN_PIN, INPUT_PULLUP);
  MPU6050_wakeup();

  // Binary Semaphore 建立
  xStartSem = xSemaphoreCreateBinary();
  xInferSem = xSemaphoreCreateBinary();

  // Task 建立（Due 96KB SRAM，stack 充裕）
  // stack 單位：words（4 bytes each on ARM Cortex-M3）
  xTaskCreate(TaskButton,    "Button", 256,  NULL, 1, NULL);
  xTaskCreate(TaskSample,    "Sample", 512,  NULL, 3, NULL);
  xTaskCreate(TaskInference, "Infer",  768,  NULL, 2, NULL);

  Serial.print("RTOS ready. WINDOW_SIZE=");
  Serial.println(WINDOW_SIZE);
  Serial.println("Push btn to start recording.");

  // 啟動 FreeRTOS Scheduler（此後 loop() 不再被呼叫）
  vTaskStartScheduler();
}

// ============================================================
// loop() — vTaskStartScheduler() 後不會執行
// 保留為空以符合 Arduino 框架要求
// ============================================================
void loop() {
  // intentionally empty
}