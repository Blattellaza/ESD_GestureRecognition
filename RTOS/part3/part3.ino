// ============================================================
// ESD Gesture Recognition — SRAM3 + Static Filter + RTOS Pipeline
//
// SRAM3 優化重點：
//   1. 完全移除 raw_samples[150][6]（節省 1800 bytes）
//      → 靜止濾波器改為錄製迴圈中即時累積 static_frame_count
//      → 每幀讀完 ax/ay/az/gx/gy/gz 後立即呼叫 isStatic()，
//        同時量化存入 nn_input，兩者並行，不需額外儲存原始值
//   2. 移除獨立 conv1_out / pool1_out / conv2_out / gap_out / dense1_out
//      改用單一 buffer_A[2400]，透過就地運算（in-place）節省 SRAM
//   3. 就地池化：pool 輸出寫入 buffer_A 後半段，避免額外緩衝
//   4. Conv2 就地運算：讀後半段、寫前半段，兩區域完全錯開無衝突
//   5. gap_out / dense1_out 借用 buffer_A 後半段空檔
//   6. getFreeRam() 監控每次錄製後的剩餘 SRAM
//
// Static Filter 整合方式（配合 SRAM3 無 raw_samples）：
//   - TaskSampling 等待按鈕期間持續呼叫 isStatic() 預熱累積 static_counter
//   - 錄製開頭重置 static_counter / prev_accel_mag，
//     確保每次錄製都是獨立乾淨的靜止判斷
//   - 錄製迴圈中每幀即時呼叫 isStatic()，累積區域變數 static_frame_count
//   - static_frame_count 傳入 InferenceMsg_t，
//     靜止比例 >= STATIC_RATIO_THRESH → TaskInference 直接輸出 "static"，跳過 CNN
//
// RTOS Pipeline：
//   TaskSampling  (Priority 3) → 等待 button semaphore → 錄製 150 筆
//                                → 送 InferenceMsg_t* 給 xQueueInference
//   TaskInference (Priority 2) → 收到樣本 → 跑 CNN → 送結果給 xQueueOutput
//   TaskOutput    (Priority 1) → 收到結果 → Serial 輸出 + LED 控制
//
// [PERF] Pipeline 效能量測（全部使用 micros()，精度 1 us）：
//
//   Sampling period 標準差：
//     每筆樣本記錄實際 micros() 時間戳 ts[i]，
//     錄製完成後計算相鄰間隔的 mean / std / max_deviation，
//     與非 RTOS 版（micros() busy-wait）比較取樣穩定度。
//     理想值：mean = 10000 us，std 越小越好。
//
//   End-to-end latency（us）：
//     t_record_start_us    → 錄製第一筆樣本前（TaskSampling）
//     t_inference_done_us  → CNN 推論完成後（TaskInference）
//     t_output_recv_us     → TaskOutput 收到結果時
//     pipeline_overhead    = end_to_end - 1500000 us（扣掉固定錄製時間）
//
//   為何全用 micros() 而非 millis()：
//     millis() 精度 1 ms，最壞誤差 ±2 ms。
//     Sampling period std 預期只有 100~500 us，millis() 完全淹沒真實數值。
//     Inference→Output queue 傳遞約 1 ms，millis() 誤差已達 ±200%。
//     micros() 精度 1 us，unsigned long 32-bit 約 71 分鐘才 overflow，
//     單次錄製完全安全。
//
//   [為何用 malloc 而非 pvPortMalloc]
//   Arduino Due FreeRTOS library 不 export pvPortMalloc 符號，
//   標準 malloc() 在 Due (newlib heap) 上完全安全可用。
// ============================================================

#include "I2C_GPIO.h"
#include "nn_ops.h"
#include <Arduino.h>
#include "src/FreeRTOS.h"
#include "src/task.h"
#include "src/queue.h"
#include "src/semphr.h"
#include <stdlib.h>   // malloc / free
#include <string.h>   // memcpy

// [RTOS] Arduino Due 上使用標準 malloc/free 取代 pvPortMalloc/pvPortFree
// 原因：Arduino Due FreeRTOS library 不 export pvPortMalloc，
//       但 malloc() 在 Due (newlib heap) 上完全安全可用。
#define RTOS_MALLOC(sz)  malloc(sz)
#define RTOS_FREE(ptr)   free(ptr)

#define I2C_use_GPIO 1

static const uint8_t MPU_ADDR = 0x68;

#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

#define BTN_PIN       12

#define DEBOUNCE_MS   30

#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  10000UL   // 100 Hz = 10000 us
#define RECORD_SECONDS      1.5
#define NUM_SAMPLES         150       // 100 Hz * 1.5 s
#define RECORD_DURATION_US  1500000UL // 150 * 10000 us，用於 pipeline overhead 計算

// [STATIC FILTER] -------------------------------------------------------
// 閾值設計依據：
//   MPU6050 noise floor ≈ 0.01g (accel), 1–3 deg/s (gyro)
//   取 noise × 5~8 倍作為安全邊界，避免雜訊觸發誤判
//   同時需要連續 STATIC_FRAMES 幀都滿足條件，防止瞬間抖動誤判
//
//   STATIC_ACCEL_THRESH: accel magnitude 相鄰幀變化量閾值 (單位: g)
//     [原始] 0.08f — 僅適用於桌上靜止（delta_mag ≈ 0.002–0.005g）
//     [修改] 0.15f — 放寬以涵蓋手持靜止（delta_mag ≈ 0.02–0.06g）
//                    動態手勢 delta_mag >> 0.15g，仍有足夠區隔
//
//   STATIC_GYRO_THRESH: gyro magnitude 閾值 (單位: deg/s)
//     [原始] 15.0f — 手持靜止時 gyro_mag 可達 3–12 deg/s，太接近邊界
//     [修改] 25.0f — 手持靜止最高約 12 deg/s，給 2x 緩衝
//                    動態手勢 gyro_mag >> 40 deg/s，仍有足夠區隔
//
//   STATIC_FRAMES: 需連續幾幀都滿足才判定為靜止
//     @ 100 Hz，10 幀 = 100 ms，足夠過濾瞬間抖動（維持不變）
// -----------------------------------------------------------------------
#define STATIC_ACCEL_THRESH  0.15f   // g      [修改: 0.08 → 0.15，涵蓋手持靜止]
#define STATIC_GYRO_THRESH   25.0f   // deg/s  [修改: 15.0 → 25.0，涵蓋手持靜止]
#define STATIC_FRAMES        10      // 連續幀數（維持不變）

// [STATIC FILTER] 靜止幀比例閾值
// 整段錄製中，靜止幀比例 >= 此值 → 判定整段為 static，跳過 CNN
// [修改] 0.70 → 0.60：手持時偶發超標幀較多，放寬比例門檻更穩健
#define STATIC_RATIO_THRESH  0.60f   // [修改: 0.70 → 0.60]

// [STATIC FILTER] 全域狀態變數
static float prev_accel_mag = 1.0f;  // 初始化為 1g（靜止重力）
static int   static_counter = 0;     // 連續靜止幀計數器
// -----------------------------------------------------------------------

uint32_t sampleId = 0;

// ============================================================
// [SRAM3] 緩衝區配置
//
// 完全移除：
//   int16_t raw_samples[150][6]  → 1800 bytes  省去
//                                   靜止濾波改為錄製迴圈中即時累積
//   int8_t  conv1_out[150][16]   → 2400 bytes
//   int8_t  pool1_out[75][16]    → 1200 bytes   全部合併為
//   int8_t  conv2_out[75][32]    → 2400 bytes   buffer_A[2400]
//   int8_t  gap_out[32]          →   32 bytes   就地運算
//   int8_t  dense1_out[16]       →   16 bytes   省去 6048 bytes
//
// 保留：
//   int8_t  nn_input[150][6]     →  900 bytes（量化後輸入，推論用）
//   int8_t  buffer_A[2400]       → 2400 bytes（單一共用推論緩衝）
//   int8_t  logits[3]            →    3 bytes（最終輸出）
// ============================================================
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];  //  900 bytes：量化後 NN 輸入
#define LAYER_BUFFER_SIZE 2400
int8_t buffer_A[LAYER_BUFFER_SIZE];           // 2400 bytes：推論共用緩衝
int8_t logits[NN_NUM_CLASSES];                //    3 bytes：最終分類 logits

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

// ============================================================
// [RTOS] 訊息結構定義
// ============================================================

// [RTOS] Sampling → Inference
// [PERF] t_record_start_us：micros()，精度 1 us
//   用於計算 end-to-end latency
typedef struct {
  int8_t        nn_input[NN_INPUT_LEN][NN_INPUT_CH];  // 量化後輸入（900 bytes）
  int           static_frame_count;                    // 本次錄製靜止幀數
  uint32_t      sample_id;                             // 樣本編號（供 log 用）
  unsigned long t_record_start_us;                     // [PERF] 錄製開始時間（micros）
  // === 新增：latency timing ===
  unsigned long t_last_sample_us;                      // [PERF] 最後一筆取樣完成後的時間戳（micros）
} InferenceMsg_t;

// [RTOS] Inference → Output
// [PERF] 兩個時間戳都用 micros()，TaskOutput 收到後計算各段延遲
typedef struct {
  int           result_idx;            // -1: static, 0~(NN_NUM_CLASSES-1): gesture
  float         static_ratio;          // 靜止幀比例（供 log 用）
  uint32_t      sample_id;             // 樣本編號（供 log 用）
  unsigned long t_record_start_us;     // [PERF] 從 Sampling 帶過來（micros）
  // === 新增：latency timing ===
  unsigned long t_last_sample_us;      // [PERF] 最後一筆取樣完成後的時間戳（micros）
  unsigned long t_inference_done_us;   // [PERF] Inference 完成時間（micros）
} OutputMsg_t;

// ============================================================
// [RTOS] 核心物件
// ============================================================

// xQueueInference: 傳遞 InferenceMsg_t* 指標（4 bytes），深度 1
// xQueueOutput:    傳遞 OutputMsg_t 值，深度 2
static QueueHandle_t     xQueueInference = NULL;
static QueueHandle_t     xQueueOutput    = NULL;

// xSemaphoreBtn：button ISR → TaskSampling
static SemaphoreHandle_t xSemaphoreBtn   = NULL;

// Task handles（除錯用）
static TaskHandle_t hTaskSampling  = NULL;
static TaskHandle_t hTaskInference = NULL;
static TaskHandle_t hTaskOutput    = NULL;

// ============================================================
// 函式宣告
// ============================================================
void    MPU6050_wakeup();
bool    readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                     int16_t *gx, int16_t *gy, int16_t *gz);
bool    isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
                 int16_t gx_raw, int16_t gy_raw, int16_t gz_raw);
int8_t  quantize_single_axis(int16_t raw_val, int32_t s,
                              int32_t z, uint8_t shift);
void    btnISR();
void    TaskSampling (void *pvParameters);
void    TaskInference(void *pvParameters);
void    TaskOutput   (void *pvParameters);

// ============================================================
// [SRAM3] 適用於 Arduino Due (SAM3X8E) 的剩餘 SRAM 檢查函式
// 原理：stack top（區域變數位址）減去 heap top（sbrk(0)）
// ============================================================
extern "C" char* sbrk(int incr);
void getFreeRam() {
  char top;
  Serial.print("Free SRAM: ");
  Serial.print(&top - reinterpret_cast<char*>(sbrk(0)));
  Serial.println(" bytes");
}

// ============================================================
// Part1: I2C Communication and Data Collection
// ============================================================

// Todo 1.1: Wake up MPU6050 by writing to its power management register.
// Requirements:
// 1) Must use your GPIO-based I2C functions (NO Wire.h).
// 2) Must perform a valid I2C register write transaction:
//    - START
//    - SLA+W
//    - register address
//    - data
//    - STOP
// Notes:
// - MPU6050 PWR_MGMT_1 register address: 0x6B
// - Wakeup value: 0x00 (clear sleep bit)
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
// Todo 1.2: Read raw accelerometer + gyroscope data from MPU6050.
//
// Transaction flow:
//
//   START
//   SLA+W
//   register address (ACCEL_XOUT_H)
//   REPEATED START
//   SLA+R
//   read ax_H, ax_L     (ACK, ACK)   buf[0..1]
//   read ay_H, ay_L     (ACK, ACK)   buf[2..3]
//   read az_H, az_L     (ACK, ACK)   buf[4..5]
//   read temp_H, temp_L (ACK, ACK)   buf[6..7]  (discarded)
//   read gx_H, gx_L     (ACK, ACK)   buf[8..9]
//   read gy_H, gy_L     (ACK, ACK)   buf[10..11]
//   read gz_H           (ACK)        buf[12]
//   read gz_L           (NACK)       buf[13]
//   STOP
//
// The function returns true if the transaction succeeds.
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
    buf[i] = I2C_read_byte(true);   // ACK
  }
  buf[13] = I2C_read_byte(false);   // NACK (last byte)
  I2C_stop();

  *ax = (int16_t)((buf[0]  << 8) | buf[1]);
  *ay = (int16_t)((buf[2]  << 8) | buf[3]);
  *az = (int16_t)((buf[4]  << 8) | buf[5]);
  // buf[6..7] = temperature, discarded
  *gx = (int16_t)((buf[8]  << 8) | buf[9]);
  *gy = (int16_t)((buf[10] << 8) | buf[11]);
  *gz = (int16_t)((buf[12] << 8) | buf[13]);

  return true;
}

// ============================================================
// [SRAM3] quantize_single_axis()
// 即時將單軸 int16_t 原始值量化為 int8_t，供 nn_input 使用
// 公式：q = (raw_val * s + z) >> shift，並 clamp 至 [-128, 127]
// 參數來源：nn_weights.h 中的 PRE_S[], PRE_Z[], PRE_SHIFT[]
// ============================================================
int8_t quantize_single_axis(int16_t raw_val, int32_t s,
                             int32_t z, uint8_t shift) {
  int32_t q = ((int32_t)raw_val * s + z) >> shift;
  if (q >  127) q =  127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// [STATIC FILTER] -------------------------------------------------------
// isStatic(): 判斷當前 IMU 讀值是否為靜止狀態
//
// 機制（雙重條件，AND 邏輯）：
//   條件 1 — Accel magnitude 變化量 < STATIC_ACCEL_THRESH
//   條件 2 — Gyro magnitude < STATIC_GYRO_THRESH
//   兩個條件同時滿足 → static_counter++
//   任一條件不滿足  → static_counter = 0
//   static_counter >= STATIC_FRAMES → 回傳 true
// -----------------------------------------------------------------------
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
// [RTOS] Button ISR
// ============================================================
void btnISR() {
  static unsigned long lastISRus = 0;
  unsigned long now = micros();

  // [DEBOUNCE] 改用 micros() 計算間隔，與其他量測單位一致
  if ((now - lastISRus) < (unsigned long)DEBOUNCE_MS * 1000UL) return;
  lastISRus = now;

  if (digitalRead(BTN_PIN) == LOW) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(xSemaphoreBtn, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
  }
}

// ============================================================
// [RTOS] Task 1: TaskSampling  (Priority 3 — 最高)
//
// [PERF] 量測項目：
//   ts[i]：每筆樣本的實際 micros() 時間戳
//   錄製完成後計算：
//     mean period   = 平均取樣間隔（理想 10000 us）
//     std  period   = 取樣間隔標準差（越小越穩定）
//     max deviation = 最大單筆偏差（worst case jitter）
//   t_record_start_us：第一筆取樣前的 micros()，帶入 InferenceMsg_t
//
// Priority: 3（最高，確保取樣不被推論搶佔）
// ============================================================
void TaskSampling(void *pvParameters) {
  (void)pvParameters;

  for (;;) {
    // -------------------------------------------------------
    // [WAIT] 等待 button 按下（semaphore），阻塞期間讓出 CPU
    // 同時持續呼叫 isStatic() 預熱 static_counter
    // -------------------------------------------------------
    while (xSemaphoreTake(xSemaphoreBtn, pdMS_TO_TICKS(10)) == pdFALSE) {
      int16_t ax, ay, az, gx, gy, gz;
      if (readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz)) {
        isStatic(ax, ay, az, gx, gy, gz);
      }
    }

    // -------------------------------------------------------
    // [WAIT RELEASE] 等待按鈕放開，避免錄到按下瞬間的靜止資料
    // -------------------------------------------------------
    while (digitalRead(BTN_PIN) == LOW) {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
    vTaskDelay(pdMS_TO_TICKS(DEBOUNCE_MS));

    // -------------------------------------------------------
    // [ALLOC] 動態配置推論訊息緩衝區
    // -------------------------------------------------------
    InferenceMsg_t *pMsg = (InferenceMsg_t *)RTOS_MALLOC(sizeof(InferenceMsg_t));
    if (pMsg == NULL) {
      Serial.println("# ERROR: malloc failed, skip.");
      continue;
    }

    // -------------------------------------------------------
    // [RESET] 每次錄製前重置 static filter 狀態
    // -------------------------------------------------------
    static_counter           = 0;
    prev_accel_mag           = 1.0f;
    pMsg->static_frame_count = 0;
    pMsg->sample_id          = sampleId;

    Serial.println("# GET_READY");
    vTaskDelay(pdMS_TO_TICKS(1000));

    Serial.println("# START");
    Serial.println("sample_id,timestep,t_us,ax,ay,az,gx,gy,gz");

    // -------------------------------------------------------
    // [PERF] 記錄錄製開始時間（micros），帶入 InferenceMsg_t
    // 時間點：GET_READY 結束、第一筆樣本取樣前
    // -------------------------------------------------------
    pMsg->t_record_start_us = micros();

    // -------------------------------------------------------
    // [PERF] ts[]：記錄每筆樣本的實際 micros() 時間戳
    // 150 * 4 bytes = 600 bytes，放在 task stack（1024 words = 4096 bytes）
    // -------------------------------------------------------
    unsigned long ts[NUM_SAMPLES];

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xInterval = pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ);  // 10 ms

    bool record_ok = true;

    for (int i = 0; i < NUM_SAMPLES; i++) {
      // [TIMING] 精確等待到下一個 10ms 週期
      vTaskDelayUntil(&xLastWakeTime, xInterval);

      // [PERF] vTaskDelayUntil 返回後立即記錄，最接近真實取樣時刻
      ts[i] = micros();

      // TODO 2.2: Declare six int16_t variables: ax, ay, az, gx, gy, gz.
      /* Your implementation */
      int16_t ax, ay, az, gx, gy, gz;

      bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

      if (!ok) {
        Serial.print("# ERROR at timestep ");
        Serial.println(i);
        record_ok = false;
        break;
      }

      // TODO 2.3: 即時量化存入 nn_input，不再儲存 raw_samples
      /* Your implementation */
      pMsg->nn_input[i][0] = quantize_single_axis(ax, PRE_S[0], PRE_Z[0], PRE_SHIFT[0]);
      pMsg->nn_input[i][1] = quantize_single_axis(ay, PRE_S[1], PRE_Z[1], PRE_SHIFT[1]);
      pMsg->nn_input[i][2] = quantize_single_axis(az, PRE_S[2], PRE_Z[2], PRE_SHIFT[2]);
      pMsg->nn_input[i][3] = quantize_single_axis(gx, PRE_S[3], PRE_Z[3], PRE_SHIFT[3]);
      pMsg->nn_input[i][4] = quantize_single_axis(gy, PRE_S[4], PRE_Z[4], PRE_SHIFT[4]);
      pMsg->nn_input[i][5] = quantize_single_axis(gz, PRE_S[5], PRE_Z[5], PRE_SHIFT[5]);

      if (isStatic(ax, ay, az, gx, gy, gz)) {
        pMsg->static_frame_count++;
      }

      // Print CSV row
      Serial.print(sampleId);  Serial.print(',');
      Serial.print(i);         Serial.print(',');
      Serial.print(ts[i]);     Serial.print(',');
      Serial.print(ax);        Serial.print(',');
      Serial.print(ay);        Serial.print(',');
      Serial.print(az);        Serial.print(',');
      Serial.print(gx);        Serial.print(',');
      Serial.print(gy);        Serial.print(',');
      Serial.println(gz);
    }

    if (!record_ok) {
      RTOS_FREE(pMsg);
      continue;
    }

    // === 新增：latency timing — 最後一筆取樣完成後立即記錄 ===
    pMsg->t_last_sample_us = micros();

    Serial.println(micros());

    // -------------------------------------------------------
    // [PERF] 計算 sampling period 統計數據
    //
    // period[i] = ts[i] - ts[i-1]，共 NUM_SAMPLES-1 = 149 個間隔
    // 理想值：mean = 10000 us（100 Hz）
    //
    // 與非 RTOS 版比較說明：
    //   非 RTOS 版（micros() busy-wait）：
    //     CPU 全速等待，不讓出，理論上 jitter 極小（< 10 us）
    //     但 Serial.print 在迴圈內造成不規則延遲，std 可能反而偏大
    //   RTOS 版（vTaskDelayUntil）：
    //     context switch overhead 約數十 us，std 預期 50~300 us
    //     但 mean 更準確，長期不累積漂移（vTaskDelayUntil 補償執行時間）
    //     TaskSampling Priority=3 最高，被搶佔機率極低
    // -------------------------------------------------------
    {
      float sum_p  = 0.0f;
      float sum_p2 = 0.0f;
      int   n      = NUM_SAMPLES - 1;  // 149 個間隔

      for (int i = 1; i < NUM_SAMPLES; i++) {
        float period = (float)(ts[i] - ts[i - 1]);
        sum_p  += period;
        sum_p2 += period * period;
      }

      float mean_p = sum_p / (float)n;
      // 變異數：E[X²] - (E[X])²
      float var_p  = (sum_p2 / (float)n) - (mean_p * mean_p);
      float std_p  = sqrtf(var_p < 0.0f ? 0.0f : var_p);

      float max_dev = 0.0f;
      for (int i = 1; i < NUM_SAMPLES; i++) {
        float dev = fabsf((float)(ts[i] - ts[i - 1]) - (float)SAMPLE_INTERVAL_US);
        if (dev > max_dev) max_dev = dev;
      }

      Serial.println("# [PERF] --- Sampling Period Statistics ---");
      Serial.print("# [PERF] Ideal period:  ");
      Serial.print((float)SAMPLE_INTERVAL_US, 1);
      Serial.println(" us");
      Serial.print("# [PERF] Mean period:   ");
      Serial.print(mean_p, 1);
      Serial.println(" us");
      Serial.print("# [PERF] Std  period:   ");
      Serial.print(std_p, 1);
      Serial.println(" us");
      Serial.print("# [PERF] Max deviation: ");
      Serial.print(max_dev, 1);
      Serial.println(" us");
    }

    // -------------------------------------------------------
    // [SEND] 送樣本給 TaskInference（傳指標，避免複製 900 bytes）
    // TaskInference 負責 RTOS_FREE(pMsg)
    // -------------------------------------------------------
    xQueueSend(xQueueInference, &pMsg, portMAX_DELAY);

    sampleId++;
  }
}

// ============================================================
// [RTOS] Task 2: TaskInference  (Priority 2 — 中)
//
// [PERF] 量測項目：
//   t0_us：收到 pMsg 後立即記錄，代表純 CNN 推論開始時間
//   t_inference_done_us：argmax 完成後立即記錄，代表推論結束時間
//   純 CNN 推論耗時 = t_inference_done_us - t0_us（us）
//
// Priority: 2
// ============================================================
void TaskInference(void *pvParameters) {
  (void)pvParameters;

  for (;;) {
    InferenceMsg_t *pMsg = NULL;
    xQueueReceive(xQueueInference, &pMsg, portMAX_DELAY);
    if (pMsg == NULL) continue;

    // [PERF] 純 CNN 推論開始時間（micros）
    unsigned long t0_us = micros();

    // -------------------------------------------------------
    // [STATIC FILTER]
    // static_frame_count 已在錄製迴圈中即時累積，直接使用
    // -------------------------------------------------------
    float static_ratio = (float)pMsg->static_frame_count / (float)NUM_SAMPLES;

    OutputMsg_t outMsg;
    outMsg.sample_id         = pMsg->sample_id;
    outMsg.static_ratio      = static_ratio;
    outMsg.t_record_start_us = pMsg->t_record_start_us;  // [PERF] 帶過來
    // === 新增：latency timing ===
    outMsg.t_last_sample_us  = pMsg->t_last_sample_us;   // [PERF] 帶過來

    if (static_ratio >= STATIC_RATIO_THRESH) {
      outMsg.result_idx          = -1;
      // === 新增：latency timing — static 分支也記錄推論完成時間 ===
      outMsg.t_inference_done_us = micros();  // [PERF]
      RTOS_FREE(pMsg);
      xQueueSend(xQueueOutput, &outMsg, portMAX_DELAY);
      continue;
    }

    // TODO 2.4: Preprocess raw_samples into nn_input using preprocess_input().
    /* Your implementation */
    // [SRAM3] 已在錄製迴圈中即時量化，直接 memcpy 到全域 nn_input
    memcpy(nn_input, pMsg->nn_input, sizeof(nn_input));

    // [FREE] 複製完畢，立即釋放
    RTOS_FREE(pMsg);
    pMsg = NULL;

    // -------------------------------------------------------
    // [CONV1] 1D Convolution (kernel=3, pad=1)
    // 輸入: nn_input  [150][6]  → flat
    // 輸出: buffer_A[0..2399]   150 x 16 = 2400 bytes
    // -------------------------------------------------------
    conv1d(
      (const int8_t *)nn_input,
      (int8_t *)buffer_A,
      CONV1_W, CONV1_B,
      NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH,
      RQ_MULT_CONV1, RQ_SHIFT_CONV1
    );
    relu((int8_t *)buffer_A, NN_INPUT_LEN * NN_CONV1_OUT_CH);

    // -------------------------------------------------------
    // [POOL1] Max Pooling stride=2
    // 輸入: buffer_A[0..2399]      前段 150 x 16
    // 輸出: buffer_A[1200..2399]   後段  75 x 16
    // -------------------------------------------------------
    maxpool1d(
      (const int8_t (*)[NN_CONV1_OUT_CH])(buffer_A),
      (int8_t (*)[NN_CONV1_OUT_CH])(buffer_A + 1200),
      NN_INPUT_LEN
    );

    // -------------------------------------------------------
    // [CONV2] 1D Convolution (kernel=3, pad=1)
    // 輸入: buffer_A[1200..2399]   後段  75 x 16
    // 輸出: buffer_A[0..2399]      前段  75 x 32
    // -------------------------------------------------------
    conv1d(
      (const int8_t *)(buffer_A + 1200),
      (int8_t *)buffer_A,
      CONV2_W, CONV2_B,
      NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
      RQ_MULT_CONV2, RQ_SHIFT_CONV2
    );
    relu((int8_t *)buffer_A, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

    // -------------------------------------------------------
    // [GAP] Global Average Pooling
    // 輸入: buffer_A[0..2399]   75 x 32
    // 輸出: gap_tmp[32] 暫存後寫回 buffer_A[0..31]
    // -------------------------------------------------------
    {
      int8_t gap_tmp[NN_CONV2_OUT_CH];
      global_avg_pool(
        (const int8_t *)buffer_A, gap_tmp,
        NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
        RQ_MULT_GAP, RQ_SHIFT_GAP
      );
      memcpy(buffer_A, gap_tmp, NN_CONV2_OUT_CH);
    }

    // -------------------------------------------------------
    // [DENSE1] Fully Connected
    // 輸入: buffer_A[0..31]    32 bytes
    // 輸出: buffer_A[32..47]   16 bytes
    // -------------------------------------------------------
    dense(
      (const int8_t *)buffer_A,
      (int8_t *)(buffer_A + 32),
      DENSE1_W, DENSE1_B,
      NN_CONV2_OUT_CH, NN_DENSE1_OUT,
      RQ_MULT_DENSE1, RQ_SHIFT_DENSE1
    );
    relu((int8_t *)(buffer_A + 32), NN_DENSE1_OUT);

    // -------------------------------------------------------
    // [DENSE2 / LOGITS] Fully Connected (no activation)
    // 輸入: buffer_A[32..47]   16 bytes
    // 輸出: logits[0..2]        3 bytes
    // -------------------------------------------------------
    dense(
      (const int8_t *)(buffer_A + 32),
      logits,
      DENSE2_W, DENSE2_B,
      NN_DENSE1_OUT, NN_NUM_CLASSES,
      RQ_MULT_DENSE2, RQ_SHIFT_DENSE2
    );

    // [ARGMAX]
    int pred = 0;
    for (int k = 1; k < NN_NUM_CLASSES; k++) {
      if (logits[k] > logits[pred]) pred = k;
    }

    // [PERF] 推論完成時間（micros）
    outMsg.t_inference_done_us = micros();
    outMsg.result_idx          = pred;

    xQueueSend(xQueueOutput, &outMsg, portMAX_DELAY);

    getFreeRam();

    // [PERF] 純 CNN 推論耗時（us）
    Serial.print("# [PERF] Inference latency: ");
    Serial.print(outMsg.t_inference_done_us - t0_us);
    Serial.println(" us");
  }
}

// ============================================================
// [RTOS] Task 3: TaskOutput  (Priority 1 — 最低)
//
// [PERF] 量測項目（全部 us）：
//
//   三個時間點：
//     t_record_start_us    → 錄製開始（TaskSampling）
//     t_inference_done_us  → 推論完成（TaskInference）
//     t_output_recv_us     → 輸出接收（TaskOutput，此處）
//
//   三段延遲：
//     sampling_to_inference = t_inference_done_us - t_record_start_us
//       = 錄製(~1500000 us) + CNN 推論
//     inference_to_output   = t_output_recv_us - t_inference_done_us
//       = xQueueSend + xQueueReceive 傳遞延遲
//     end_to_end            = t_output_recv_us - t_record_start_us
//       = 完整 pipeline 總延遲
//     pipeline_overhead     = end_to_end - RECORD_DURATION_US
//       = 扣掉固定錄製時間後的純 overhead
//         主要組成：CNN 推論 + queue 傳遞
//
// Priority: 1（最低，輸出不影響取樣與推論）
// ============================================================
void TaskOutput(void *pvParameters) {
  (void)pvParameters;

  for (;;) {
    OutputMsg_t msg;
    xQueueReceive(xQueueOutput, &msg, portMAX_DELAY);

    // [PERF] 收到結果的時間（micros），立即記錄，減少後續 Serial.print 的干擾
    unsigned long t_output_recv_us = micros();

    // [OUTPUT] 輸出結果標籤
    if (msg.result_idx == -1) {
      Serial.println("# INFERENCE_RESULT: static");
      Serial.print("# static_ratio: ");
      Serial.println(msg.static_ratio);
    } else if (msg.result_idx >= 0 && msg.result_idx < NN_NUM_CLASSES) {
      Serial.print("# INFERENCE_RESULT: ");
      Serial.println(LABEL_NAMES[msg.result_idx]);
    } else {
      Serial.println("# INFERENCE_RESULT: unknown");
    }

    // -------------------------------------------------------
    // [PERF] Pipeline Latency Report（全部以 us 計算）
    // -------------------------------------------------------
    unsigned long sampling_to_inference = msg.t_inference_done_us
                                          - msg.t_record_start_us;
    unsigned long inference_to_output   = t_output_recv_us
                                          - msg.t_inference_done_us;
    unsigned long end_to_end            = t_output_recv_us
                                          - msg.t_record_start_us;
    unsigned long pipeline_overhead     = (end_to_end > RECORD_DURATION_US)
                                          ? (end_to_end - RECORD_DURATION_US)
                                          : 0UL;

    Serial.println("# [PERF] --- Pipeline Latency Report ---");
    Serial.print("# [PERF] Sampling→Inference done: ");
    Serial.print(sampling_to_inference);
    Serial.println(" us  (includes ~1500000 us recording)");
    Serial.print("# [PERF] Inference→Output recv:   ");
    Serial.print(inference_to_output);
    Serial.println(" us  (queue transfer latency)");
    Serial.print("# [PERF] End-to-end total:        ");
    Serial.print(end_to_end);
    Serial.println(" us");
    Serial.print("# [PERF] Pipeline overhead:       ");
    Serial.print(pipeline_overhead);
    Serial.println(" us  (end_to_end - 1500000 us recording)");

    // === 新增：latency timing — [RESULT] 機器可解析格式 ===
    // 格式：[RESULT] trial=N first=T1 last=T2 pred=T3
    //   trial : 本次錄製序號（sampleId）
    //   first : 第一筆取樣前的時間戳（TaskSampling 開始錄製）
    //   last  : 最後一筆取樣完成後的時間戳
    //   pred  : TaskInference 推論完成後的時間戳
    // analyze_log.py 會解析此行計算 end-to-end latency
    Serial.print("[RESULT] trial=");
    Serial.print(msg.sample_id);
    Serial.print(" first=");
    Serial.print(msg.t_record_start_us);
    Serial.print(" last=");
    Serial.print(msg.t_last_sample_us);
    Serial.print(" pred=");
    Serial.println(msg.t_inference_done_us);
    // ========================================================

    Serial.println("# END");
  }
}

// ============================================================
// setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Dataset collector ready.");
  Serial.println("Push btn to start recording.");

  pinMode(BTN_PIN, INPUT_PULLUP);

  MPU6050_wakeup();

  xSemaphoreBtn   = xSemaphoreCreateBinary();
  xQueueInference = xQueueCreate(1, sizeof(InferenceMsg_t *));
  xQueueOutput    = xQueueCreate(2, sizeof(OutputMsg_t));

  attachInterrupt(digitalPinToInterrupt(BTN_PIN), btnISR, FALLING);

  // TaskSampling  stack=1024 words：容納 ts[150](600B) + readImu6_raw buf[14] + isStatic float
  // TaskInference stack=512  words：大陣列全為全域，stack 只需 gap_tmp[32] 與迴圈變數
  // TaskOutput    stack=256  words：只做 Serial.print + LED
  xTaskCreate(TaskSampling,  "Sampling",  1024, NULL, 3, &hTaskSampling);
  xTaskCreate(TaskInference, "Inference", 512,  NULL, 2, &hTaskInference);
  xTaskCreate(TaskOutput,    "Output",    256,  NULL, 1, &hTaskOutput);

  vTaskStartScheduler();

  Serial.println("# ERROR: Scheduler failed to start!");
  for (;;) {}
}

// ============================================================
// loop(): FreeRTOS Scheduler 啟動後不會被呼叫
// ============================================================
void loop() {}
