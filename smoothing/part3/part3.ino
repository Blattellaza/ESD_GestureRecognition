// ============================================================
// ESD Gesture Recognition — SRAM3 + Static Filter + Moving Average Filter
//
// 本版本基於 prefilter_with_SRAM3，新增 Moving Average Filter（任務二）：
//   - 對 ax、ay、az 三軸加速度計原始值施加 Circular Buffer MA 濾波
//   - WINDOW_SIZE 可切換 1 / 3 / 5，控制濾波強度與手勢響應速度
//   - 使用 circular buffer 實作，避免陣列整體位移，省 SRAM
//   - WINDOW_SIZE=1 等同無濾波，可作為對照組
//
// Moving Average Filter 設計原理：
//   window 越大 → 雜訊越平滑 → sampling period std 越小（stability 越好）
//   window 越大 → 新訊號反應越慢 → gesture response latency 越高
//   預期甜蜜點約 window=3（平滑效果明顯，latency 增加僅 20ms @100Hz）
//
// SRAM 消耗（3 軸 × 1 個 MAFilter）：
//   每個 MAFilter: buf[WINDOW_SIZE×2] + head(2) + count(2) + sum(4) bytes
//   WINDOW_SIZE=5: 3 × 18 = 54 bytes — 對 SRAM3 預算影響極小
//
// 新增量測指標（與 analyze_log.py 對應）：
//   [RESULT] trial=N first=T1 last=T2 pred=T3  — end-to-end latency
//   [SMOOTH] window=N mean=X std=X max_dev=X   — 取樣穩定度統計
//
// 本版版本不使用 FreeRTOS，為序列執行。
// ============================================================

#include "I2C_GPIO.h"
#include "nn_ops.h"
#include <string.h>   // memset

#define I2C_use_GPIO 1

static const uint8_t MPU_ADDR = 0x68;

#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

#define BTN_PIN       12

#define DEBOUNCE_MS   30

#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  10000UL   // 100 Hz = 10 ms
#define RECORD_SECONDS      1.5
#define NUM_SAMPLES         150       // 100 Hz * 1.5 s

// ============================================================
// === 新增：Moving Average Filter 設定 ===
// 修改此值即可切換濾波強度：1（無濾波）/ 3（推薦）/ 5（強濾波）
// 注意：修改後須重新編譯燒錄
// ============================================================
#define WINDOW_SIZE 5

// MAFilter: circular buffer 實作的移動平均濾波器
// 用於 ax / ay / az 三軸，在進入 NN pipeline 前降噪
//
// 欄位說明：
//   buf[]  — 環形儲存槽（儲存最近 WINDOW_SIZE 筆原始值）
//   head   — 下一筆寫入的位置（0 ~ WINDOW_SIZE-1）
//   count  — 目前有效資料筆數（< WINDOW_SIZE 時為暖機期）
//   sum    — 目前有效資料的累積和（避免每次重新加總）
typedef struct {
  int16_t buf[WINDOW_SIZE];
  int     head;
  int     count;
  int32_t sum;
} MAFilter;

// 三軸加速度計各自的濾波器狀態（全域，每次錄製前重置）
static MAFilter ma_ax;
static MAFilter ma_ay;
static MAFilter ma_az;

// ma_update(): 更新 circular buffer，回傳新的移動平均值
// 參數: f = 濾波器狀態指標; val = 新的原始取樣值（int16_t ADC counts）
// 回傳: 最近 count 筆的整數平均值（暖機期內 count < WINDOW_SIZE）
static int16_t ma_update(MAFilter *f, int16_t val) {
  // 若 buffer 已滿，先從 sum 扣掉最舊的值
  if (f->count == WINDOW_SIZE) {
    f->sum -= (int32_t)f->buf[f->head];
  } else {
    f->count++;  // 暖機期：尚未填滿，count 遞增
  }
  // 寫入新值並推進指標
  f->buf[f->head] = val;
  f->sum += (int32_t)val;
  f->head = (f->head + 1) % WINDOW_SIZE;
  // 整數除法，不引入浮點數（量化前的 raw 值仍為整數）
  return (int16_t)(f->sum / f->count);
}
// ============================================================

// [STATIC FILTER] -------------------------------------------------------
#define STATIC_ACCEL_THRESH  0.15f   // g
#define STATIC_GYRO_THRESH   25.0f   // deg/s
#define STATIC_FRAMES        10

#define STATIC_RATIO_THRESH  0.60f

static float prev_accel_mag = 1.0f;
static int   static_counter = 0;
// -----------------------------------------------------------------------

bool isRecording = false;

unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;

uint32_t sampleId = 0;

// ============================================================
// [SRAM3] 緩衝區配置（與 prefilter_with_SRAM3 相同）
// MAFilter 僅需 54 bytes（WINDOW_SIZE=5 時），不影響主要預算
// ============================================================
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];  //  900 bytes
#define LAYER_BUFFER_SIZE 2400
int8_t buffer_A[LAYER_BUFFER_SIZE];           // 2400 bytes
int8_t logits[NN_NUM_CLASSES];                //    3 bytes

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

// ============================================================
// 函式宣告
// ============================================================
void    MPU6050_wakeup();
bool    readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                     int16_t *gx, int16_t *gy, int16_t *gz);
void    handleButton();
void    recordOneSample();
void    runInference(int static_frame_count,
                     unsigned long t_first_sample_us,
                     unsigned long t_last_sample_us);
bool    isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
                 int16_t gx_raw, int16_t gy_raw, int16_t gz_raw);
int8_t  quantize_single_axis(int16_t raw_val, int32_t s,
                              int32_t z, uint8_t shift);

extern "C" char* sbrk(int incr);
void getFreeRam() {
  char top;
  Serial.print("Free SRAM: ");
  Serial.print(&top - reinterpret_cast<char*>(sbrk(0)));
  Serial.println(" bytes");
}

// ============================================================
// MPU6050_wakeup()
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
// readImu6_raw()
// ============================================================
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz) {
  uint8_t buf[14];

  I2C_start();
  if (!I2C_write_byte(MPU_ADDR << 1) ||
      !I2C_write_byte(ACCEL_XOUT_H)) {
    I2C_stop();
    return false;
  }

  I2C_repeated_start();
  if (!I2C_write_byte((MPU_ADDR << 1) | 1)) {
    I2C_stop();
    return false;
  }

  for (uint8_t i = 0; i < 13; i++) {
    buf[i] = I2C_read_byte(true);
  }
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
// quantize_single_axis()
// ============================================================
int8_t quantize_single_axis(int16_t raw_val, int32_t s, int32_t z, uint8_t shift) {
  int32_t q = ((int32_t)raw_val * s + z) >> shift;
  if (q >  127) q =  127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// ============================================================
// isStatic()
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

  float gyro_mag = sqrtf(gx_d*gx_d + gy_d*gy_d + gz_d*gz_d);

  if (delta_mag < STATIC_ACCEL_THRESH && gyro_mag < STATIC_GYRO_THRESH) {
    static_counter++;
  } else {
    static_counter = 0;
  }

  return (static_counter >= STATIC_FRAMES);
}

// ============================================================
// setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(BTN_PIN, INPUT_PULLUP);
  MPU6050_wakeup();

  Serial.print("Smoothing version ready. WINDOW_SIZE=");
  Serial.println(WINDOW_SIZE);
  Serial.println("Push btn to start recording.");
}

// ============================================================
// loop()
// ============================================================
void loop() {
  handleButton();

  int16_t ax, ay, az, gx, gy, gz;
  bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);
  if (ok) {
    isStatic(ax, ay, az, gx, gy, gz);
  }
}

// ============================================================
// handleButton()
// ============================================================
void handleButton() {
  int reading = digitalRead(BTN_PIN);

  if (reading != lastButtonReading) {
    lastDebounceMs = millis();
  }

  if ((millis() - lastDebounceMs) > DEBOUNCE_MS) {
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == LOW && !isRecording) {
        recordOneSample();
        getFreeRam();
      }
    }
  }

  lastButtonReading = reading;
}

// ============================================================
// recordOneSample()
// ============================================================
void recordOneSample() {
  isRecording = true;

  // [STATIC FILTER] 重置靜止濾波器狀態
  static_counter    = 0;
  prev_accel_mag    = 1.0f;
  int static_frame_count = 0;

  // === 新增：重置 Moving Average Filter 狀態 ===
  // 每次錄製前清空 circular buffer，確保本次錄製不受上次資料污染
  memset(&ma_ax, 0, sizeof(MAFilter));
  memset(&ma_ay, 0, sizeof(MAFilter));
  memset(&ma_az, 0, sizeof(MAFilter));
  // =============================================

  Serial.println("# GET_READY");
  Serial.print("# WINDOW_SIZE=");
  Serial.println(WINDOW_SIZE);
  delay(1000);

  Serial.println("# START");
  Serial.println("sample_id,timestep,t_us,ax_raw,ay_raw,az_raw,ax_filt,ay_filt,az_filt,gx,gy,gz");

  // === 新增：latency timing ===
  unsigned long t_first_sample_us = 0;
  unsigned long t_last_sample_us  = 0;
  // ===========================

  // === 新增：取樣間隔統計 ===
  // 記錄每筆樣本的時間戳，錄製完成後計算 mean / std / max_dev
  // 600 bytes on stack（Arduino Due 32KB stack，安全無虞）
  unsigned long ts[NUM_SAMPLES];
  // ==========================

  unsigned long nextSampleUs = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    while ((long)(micros() - nextSampleUs) < 0) {
      // busy-wait until next 10 ms slot
    }

    unsigned long tUs = micros();

    // === 新增：latency timing — 記錄首末取樣時間戳 ===
    if (i == 0)              t_first_sample_us = tUs;
    if (i == NUM_SAMPLES-1)  t_last_sample_us  = tUs;
    ts[i] = tUs;  // 記錄每筆時間戳，供穩定度統計用
    // ===================================================

    int16_t ax, ay, az, gx, gy, gz;
    bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

    if (!ok) {
      Serial.print("# ERROR at timestep ");
      Serial.println(i);
      isRecording = false;
      return;
    }

    // === 新增：Moving Average Filter ===
    // 在 ax/ay/az 進入 NN pipeline 前施加濾波
    // gyro (gx/gy/gz) 不濾波（手勢方向由旋轉速率決定，濾波會抹去方向性）
    //
    // 設計取捨：
    //   對 isStatic() 也使用濾波後的值 → 靜止判斷更穩定（雜訊不易觸發抖動）
    //   缺點：暖機期（前 WINDOW_SIZE-1 幀）的平均值偏向初始讀值
    //         錄製開頭重置 buffer，暖機影響僅 20ms (WINDOW_SIZE=3, 100Hz)
    int16_t ax_f = ma_update(&ma_ax, ax);
    int16_t ay_f = ma_update(&ma_ay, ay);
    int16_t az_f = ma_update(&ma_az, az);
    // =====================================

    // 使用濾波後的 ax_f/ay_f/az_f 量化存入 nn_input
    nn_input[i][0] = quantize_single_axis(ax_f, PRE_S[0], PRE_Z[0], PRE_SHIFT[0]);
    nn_input[i][1] = quantize_single_axis(ay_f, PRE_S[1], PRE_Z[1], PRE_SHIFT[1]);
    nn_input[i][2] = quantize_single_axis(az_f, PRE_S[2], PRE_Z[2], PRE_SHIFT[2]);
    // gyro 使用原始值（不濾波）
    nn_input[i][3] = quantize_single_axis(gx,   PRE_S[3], PRE_Z[3], PRE_SHIFT[3]);
    nn_input[i][4] = quantize_single_axis(gy,   PRE_S[4], PRE_Z[4], PRE_SHIFT[4]);
    nn_input[i][5] = quantize_single_axis(gz,   PRE_S[5], PRE_Z[5], PRE_SHIFT[5]);

    // isStatic 使用濾波後的加速度計值，gyro 保持原始
    if (isStatic(ax_f, ay_f, az_f, gx, gy, gz)) {
      static_frame_count++;
    }

    // CSV: 同時印出原始值與濾波後值，方便對比
    Serial.print(sampleId); Serial.print(',');
    Serial.print(i);        Serial.print(',');
    Serial.print(tUs);      Serial.print(',');
    Serial.print(ax);       Serial.print(',');   // ax_raw
    Serial.print(ay);       Serial.print(',');   // ay_raw
    Serial.print(az);       Serial.print(',');   // az_raw
    Serial.print(ax_f);     Serial.print(',');   // ax 濾波後
    Serial.print(ay_f);     Serial.print(',');   // ay 濾波後
    Serial.print(az_f);     Serial.print(',');   // az 濾波後
    Serial.print(gx);       Serial.print(',');
    Serial.print(gy);       Serial.print(',');
    Serial.println(gz);

    nextSampleUs += SAMPLE_INTERVAL_US;
  }

  // === 新增：取樣間隔統計 ===
  // 計算 sampling period mean / std / max_dev，評估取樣穩定性
  // 不同 WINDOW_SIZE 下的 std 可量化濾波對穩定性的影響
  {
    int   n      = NUM_SAMPLES - 1;  // 149 個間隔
    float sum_p  = 0.0f;

    for (int i = 1; i < NUM_SAMPLES; i++) {
      sum_p += (float)(ts[i] - ts[i - 1]);
    }
    float mean_p = sum_p / (float)n;

    // 兩步計算法：先算 mean，再算偏差平方和
    // 避免 E[X²]-(E[X])² 的大數相減精度損失（catastrophic cancellation）
    float var_sum = 0.0f;
    for (int i = 1; i < NUM_SAMPLES; i++) {
      float dev = (float)(ts[i] - ts[i - 1]) - mean_p;
      var_sum += dev * dev;
    }
    float std_p = sqrtf(var_sum / (float)n);

    float max_dev = 0.0f;
    for (int i = 1; i < NUM_SAMPLES; i++) {
      float dev = fabsf((float)(ts[i] - ts[i-1]) - (float)SAMPLE_INTERVAL_US);
      if (dev > max_dev) max_dev = dev;
    }

    // [SMOOTH] 格式供 analyze_log.py 解析
    Serial.print("[SMOOTH] window=");   Serial.print(WINDOW_SIZE);
    Serial.print(" mean=");             Serial.print(mean_p, 1);
    Serial.print(" std=");              Serial.print(std_p, 1);
    Serial.print(" max_dev=");          Serial.println(max_dev, 1);
  }
  // ==========================

  runInference(static_frame_count, t_first_sample_us, t_last_sample_us);

  Serial.println("# END");
  sampleId++;
  isRecording = false;
}

// ============================================================
// runInference()
// ============================================================
void runInference(int static_frame_count,
                  unsigned long t_first_sample_us,
                  unsigned long t_last_sample_us) {
  float static_ratio = (float)static_frame_count / (float)NUM_SAMPLES;

  if (static_ratio >= STATIC_RATIO_THRESH) {
    unsigned long t_inference_done_us = micros();
    Serial.println("# INFERENCE_RESULT: static");
    Serial.print("# static_ratio: ");
    Serial.println(static_ratio);
    Serial.print("[RESULT] trial=");   Serial.print(sampleId);
    Serial.print(" first=");           Serial.print(t_first_sample_us);
    Serial.print(" last=");            Serial.print(t_last_sample_us);
    Serial.print(" pred=");            Serial.println(t_inference_done_us);
    return;
  }

  unsigned long startUs = micros();

  // Step 1. Conv1 + ReLU
  conv1d(&nn_input[0][0], buffer_A,
         CONV1_W, CONV1_B,
         NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH,
         RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu(buffer_A, NN_INPUT_LEN * NN_CONV1_OUT_CH);

  // Step 2. MaxPool1 就地池化
  maxpool1d((int8_t(*)[NN_CONV1_OUT_CH])buffer_A,
            (int8_t(*)[NN_CONV1_OUT_CH])(&buffer_A[1200]),
            NN_INPUT_LEN);

  // Step 3. Conv2 就地運算 + ReLU
  conv1d(&buffer_A[1200], buffer_A,
         CONV2_W, CONV2_B,
         NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu(buffer_A, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  // Step 4. 後端記憶體借用
  int8_t* gap_out    = &buffer_A[1200];
  int8_t* dense1_out = &buffer_A[1232];

  // Step 5. Global Average Pooling
  global_avg_pool((int8_t*)buffer_A, gap_out,
                  NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // Step 6. Dense1 + ReLU
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);

  // Step 7. Dense2（logits）
  dense(dense1_out, logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  unsigned long latencyUs = micros() - startUs;

  int label = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[label]) label = i;
  }

  unsigned long t_inference_done_us = micros();

  Serial.print("# INFERENCE_LATENCY_US,");
  Serial.println(latencyUs);
  Serial.print("# PREDICTION,");
  Serial.println(LABEL_NAMES[label]);
  Serial.print("# static_ratio: ");
  Serial.println(static_ratio);

  // [RESULT] 機器可解析格式（與 prefilter_with_SRAM3 相同）
  Serial.print("[RESULT] trial=");   Serial.print(sampleId);
  Serial.print(" first=");           Serial.print(t_first_sample_us);
  Serial.print(" last=");            Serial.print(t_last_sample_us);
  Serial.print(" pred=");            Serial.println(t_inference_done_us);
}
