// ============================================================
// ESD Gesture Recognition — SRAM3 + Static Filter 完整整合版
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
//   - loop() 持續呼叫 isStatic() 預熱累積 static_counter
//   - recordOneSample() 開頭重置 static_counter / prev_accel_mag，
//     確保每次錄製都是獨立乾淨的靜止判斷
//   - 錄製迴圈中每幀即時呼叫 isStatic()，累積區域變數 static_frame_count
//   - static_frame_count 傳入 runInference()，
//     靜止比例 >= STATIC_RATIO_THRESH → 直接輸出 "static"，跳過 CNN
// ============================================================

#include "I2C_GPIO.h"
#include "nn_ops.h"

// #include <FreeRTOS.h>
// #include <task.h>

#define I2C_use_GPIO 1

static const uint8_t MPU_ADDR = 0x68;

#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

#define LED_PIN       6
#define BTN_PIN       12

#define DEBOUNCE_MS   30

#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  10000UL   // 100 Hz = 10 ms
#define RECORD_SECONDS      1.5
#define NUM_SAMPLES         150       // 100 Hz * 1.5 s

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

bool isRecording = false;

unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;

uint32_t sampleId = 0;

// ============================================================
// [SRAM3] 緩衝區配置
//
// 完全移除：
//   int16_t raw_samples[150][6]  → 1800 bytes ✅ 省去
//                                   靜止濾波改為錄製迴圈中即時累積
//   int8_t  conv1_out[150][16]   → 2400 bytes ┐
//   int8_t  pool1_out[75][16]    → 1200 bytes │ 全部合併為
//   int8_t  conv2_out[75][32]    → 2400 bytes │ buffer_A[2400]
//   int8_t  gap_out[32]          →   32 bytes │ 就地運算
//   int8_t  dense1_out[16]       →   16 bytes ┘ ✅ 省去 6048 bytes
//
// 保留：
//   int8_t  nn_input[150][6]     →  900 bytes（量化後輸入，推論用）
//   int8_t  buffer_A[2400]       → 2400 bytes（單一共用推論緩衝）
//   int8_t  logits[3]            →    3 bytes（最終輸出）
//
// 靜止濾波器記憶體：
//   static_frame_count           →    4 bytes（區域變數，stack 上，不佔全域 SRAM）
// ============================================================
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];  //  900 bytes：量化後 NN 輸入
#define LAYER_BUFFER_SIZE 2400
int8_t buffer_A[LAYER_BUFFER_SIZE];           // 2400 bytes：推論共用緩衝
// int8_t buffer_B[LAYER_BUFFER_SIZE];        // 備用第二緩衝（目前不需要）
int8_t logits[NN_NUM_CLASSES];                //    3 bytes：最終分類 logits

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

// ============================================================
// 函式宣告
// ============================================================
void MPU6050_wakeup();
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz);
void handleButton();
void recordOneSample();
// === 新增：latency timing — runInference 接收三個時間戳 ===
void runInference(int static_frame_count,
                  unsigned long t_first_sample_us,
                  unsigned long t_last_sample_us);
// ============================================================

// [STATIC FILTER] 新增函式宣告
bool isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw);

// [SRAM3] 即時量化函式宣告
int8_t quantize_single_axis(int16_t raw_val, int32_t s, int32_t z, uint8_t shift);

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
  // 對每個 write_byte 做 ACK 檢查，失敗時印出錯誤提示
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

  // Step 1: register pointer write
  I2C_start();
  if (!I2C_write_byte(MPU_ADDR << 1) ||
      !I2C_write_byte(ACCEL_XOUT_H)) {
    I2C_stop();
    return false;
  }

  // Step 2: repeated START, switch to read mode
  I2C_repeated_start();
  if (!I2C_write_byte((MPU_ADDR << 1) | 1)) {
    I2C_stop();
    return false;
  }

  // Step 3: read 14 bytes (accel + temp + gyro)
  for (uint8_t i = 0; i < 13; i++) {
    buf[i] = I2C_read_byte(true);   // ACK
  }
  buf[13] = I2C_read_byte(false);   // NACK (last byte)
  I2C_stop();

  // Step 4: combine high and low bytes into signed 16-bit integers
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
// 參數來源：nn_ops.h 中的 PRE_S[], PRE_Z[], PRE_SHIFT[]
// ============================================================
int8_t quantize_single_axis(int16_t raw_val, int32_t s, int32_t z, uint8_t shift) {
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
//     計算 accel magnitude = sqrt(ax²+ay²+az²) [g]
//     Δmag = |mag_now - mag_prev|
//     靜止時 Δmag 極小（重力向量幾乎不變）
//     動態手勢時 Δmag 會因加速度變化而顯著增大
//
//   條件 2 — Gyro magnitude < STATIC_GYRO_THRESH
//     計算 gyro_mag = sqrt(gx²+gy²+gz²) [deg/s]
//     靜止時陀螺儀幾乎不旋轉，mag ≈ 1–3 deg/s
//     動態手勢時旋轉速率遠超 25 deg/s
//
//   兩個條件同時滿足 → static_counter++
//   任一條件不滿足  → static_counter = 0（重置，防止累積誤判）
//
//   static_counter >= STATIC_FRAMES → 回傳 true（判定為靜止）
//
// 為何用雙重條件：
//   - 只看 accel：手持傾斜但不動時，重力分量穩定，可能誤判
//     但若同時有微小震動，gyro 會洩漏，雙重條件可補足
//   - 只看 gyro：手持震動但不旋轉時，gyro 小但 accel 在抖
//     雙重條件可捕捉到 accel 的抖動
// -----------------------------------------------------------------------
bool isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw) {

  // 將 raw 轉換為物理單位（與 preprocess_input 相同的轉換）
  // accel: raw / 16384.0 → g (±2g 量程)
  // gyro:  raw / 131.0   → deg/s (±250 deg/s 量程)
  float ax_g = ax_raw / 16384.0f;
  float ay_g = ay_raw / 16384.0f;
  float az_g = az_raw / 16384.0f;
  float gx_d = gx_raw / 131.0f;
  float gy_d = gy_raw / 131.0f;
  float gz_d = gz_raw / 131.0f;

  // 條件 1：計算 accel magnitude 及其與前一幀的變化量
  float accel_mag = sqrtf(ax_g*ax_g + ay_g*ay_g + az_g*az_g);
  float delta_mag = fabsf(accel_mag - prev_accel_mag);
  prev_accel_mag  = accel_mag;  // 更新前一幀 magnitude

  // 條件 2：計算 gyro magnitude
  float gyro_mag = sqrtf(gx_d*gx_d + gy_d*gy_d + gz_d*gz_d);

  // 雙重條件判斷
  if (delta_mag < STATIC_ACCEL_THRESH && gyro_mag < STATIC_GYRO_THRESH) {
    static_counter++;
  } else {
    static_counter = 0;  // 任一條件不滿足，重置計數器
  }

  // 連續 STATIC_FRAMES 幀都滿足 → 判定為靜止
  return (static_counter >= STATIC_FRAMES);
}
// -----------------------------------------------------------------------


// ============================================================
// setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);
  MPU6050_wakeup();
  analogWrite(LED_PIN, 0);

  Serial.println("Dataset collector ready.");
  Serial.println("Push btn to start recording.");
}


// ============================================================
// loop()
// ============================================================
void loop() {
  handleButton();

  // [STATIC FILTER] 在 loop() 中持續偵測靜止狀態（不依賴錄製觸發）
  // 讓 static_counter 在按鈕按下前就已累積預熱，
  // 但 recordOneSample() 開頭會重置，確保錄製段獨立判斷
  int16_t ax, ay, az, gx, gy, gz;
  bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);
  if (ok) {
    isStatic(ax, ay, az, gx, gy, gz);  // 持續更新 prev_accel_mag
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
        // [SRAM3] 每次錄製完成後印出剩餘 SRAM，方便監控記憶體使用
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

  // [STATIC FILTER] 重置靜止濾波器狀態，確保每次錄製獨立判斷
  // 避免 loop() 中累積的 static_counter 和 prev_accel_mag 污染錄製段
  static_counter    = 0;
  prev_accel_mag    = 1.0f;
  int static_frame_count = 0;  // 本次錄製靜止幀計數（區域變數，不佔全域 SRAM）

  Serial.println("# GET_READY");
  digitalWrite(LED_PIN, HIGH);
  delay(1000);

  Serial.println("# START");
  Serial.println("sample_id,timestep,t_us,ax,ay,az,gx,gy,gz");

  // === 新增：latency timing ===
  unsigned long t_first_sample_us = 0;
  unsigned long t_last_sample_us  = 0;
  // ===========================

  unsigned long nextSampleUs = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    while ((long)(micros() - nextSampleUs) < 0) {
      // wait until next 10 ms slot
    }

    unsigned long tUs = micros();
    // === 新增：latency timing — 第一筆與最後一筆取樣時間戳 ===
    if (i == 0)              t_first_sample_us = tUs;
    if (i == NUM_SAMPLES-1)  t_last_sample_us  = tUs;
    // =========================================================

    // TODO 2.2: Declare six int16_t variables: ax, ay, az, gx, gy, gz.
    //           (The call to readImu6_raw() and error handling below depend on
    //           these variables being declared here.)
    /* Your implementation */
    int16_t ax, ay, az, gx, gy, gz;

    bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

    if (!ok) {
      Serial.print("# ERROR at timestep ");
      Serial.println(i);
      digitalWrite(LED_PIN, LOW);
      isRecording = false;
      return;
    }

    // TODO 2.3: 即時量化存入 nn_input，不再儲存 raw_samples
    //           Channel mapping: 0→ax, 1→ay, 2→az (accelerometer, ±2 g range)
    //                            3→gx, 4→gy, 5→gz (gyroscope, ±250 °/s range)
    /* Your implementation */
    // [SRAM3] 不存入 raw_samples，改為直接即時量化並存入 nn_input
    nn_input[i][0] = quantize_single_axis(ax, PRE_S[0], PRE_Z[0], PRE_SHIFT[0]);
    nn_input[i][1] = quantize_single_axis(ay, PRE_S[1], PRE_Z[1], PRE_SHIFT[1]);
    nn_input[i][2] = quantize_single_axis(az, PRE_S[2], PRE_Z[2], PRE_SHIFT[2]);
    nn_input[i][3] = quantize_single_axis(gx, PRE_S[3], PRE_Z[3], PRE_SHIFT[3]);
    nn_input[i][4] = quantize_single_axis(gy, PRE_S[4], PRE_Z[4], PRE_SHIFT[4]);
    nn_input[i][5] = quantize_single_axis(gz, PRE_S[5], PRE_Z[5], PRE_SHIFT[5]);

    // [STATIC FILTER] 即時判斷靜止，原始值用完即丟，不儲存
    // 與量化並行執行，不增加任何額外記憶體
    if (isStatic(ax, ay, az, gx, gy, gz)) {
      static_frame_count++;
    }

    // Print CSV row
    Serial.print(sampleId);  Serial.print(',');
    Serial.print(i);         Serial.print(',');
    Serial.print(tUs);       Serial.print(',');
    Serial.print(ax);        Serial.print(',');
    Serial.print(ay);        Serial.print(',');
    Serial.print(az);        Serial.print(',');
    Serial.print(gx);        Serial.print(',');
    Serial.print(gy);        Serial.print(',');
    Serial.println(gz);

    nextSampleUs += SAMPLE_INTERVAL_US;
  }

  // [SRAM3] 錄製結束計時標記
  Serial.println(micros());

  // === 新增：latency timing — 傳入三個時間戳 ===
  runInference(static_frame_count, t_first_sample_us, t_last_sample_us);
  // ================================================

  // [SRAM3] # END 移至 runInference() 之後
  Serial.println("# END");
  digitalWrite(LED_PIN, LOW);
  sampleId++;
  isRecording = false;
}


// ============================================================
// runInference()
// [STATIC FILTER] 接收 static_frame_count，不再需要掃描 raw_samples
// === 新增：latency timing — 接收取樣時間戳，輸出 [RESULT] 格式 ===
// ============================================================
void runInference(int static_frame_count,
                  unsigned long t_first_sample_us,
                  unsigned long t_last_sample_us) {
  // [STATIC FILTER] -------------------------------------------------------
  // 計算靜止幀比例，決定是否跳過 CNN
  //
  // 與原版差異：
  //   原版：在此重置 static_counter，重新掃描 raw_samples 150 幀
  //   現版：static_frame_count 已在錄製迴圈中即時累積，直接使用
  //         省去二次掃描，同時省去 raw_samples 1800 bytes
  // -----------------------------------------------------------------------
  float static_ratio = (float)static_frame_count / (float)NUM_SAMPLES;

  // [STATIC FILTER] 靜止判斷：超過閾值比例直接輸出，不進 CNN
  if (static_ratio >= STATIC_RATIO_THRESH) {
    // === 新增：latency timing — static 分支也記錄推論完成時間 ===
    unsigned long t_inference_done_us = micros();
    Serial.println("# INFERENCE_RESULT: static");
    Serial.print("# static_ratio: ");
    Serial.println(static_ratio);
    Serial.print("[RESULT] trial=");   Serial.print(sampleId);
    Serial.print(" first=");           Serial.print(t_first_sample_us);
    Serial.print(" last=");            Serial.print(t_last_sample_us);
    Serial.print(" pred=");            Serial.println(t_inference_done_us);
    // ===========================================================
    return;
  }

  // -----------------------------------------------------------------------
  // 以下為 CNN inference pipeline（靜止判斷未觸發才執行）
  // -----------------------------------------------------------------------

  // TODO 2.4: Preprocess raw_samples into nn_input using preprocess_input().
  /* Your implementation */
  // [SRAM3] 已在 recordOneSample() 中即時量化，此處不再呼叫 preprocess_input()
  // preprocess_input(raw_samples, nn_input);

  unsigned long startUs = micros();

  // -----------------------------------------------------------------------
  // [SRAM3] 單緩衝就地推論（buffer_A 佈局示意）
  //
  //  步驟 1 — Conv1 輸出：
  //    buffer_A[0 ~ 2399]  ← conv1_out[150][16]  (2400 bytes)
  //
  //  步驟 2 — MaxPool1 就地池化：
  //    讀取 buffer_A[0 ~ 2399]（conv1_out）
  //    寫入 buffer_A[1200 ~ 2399]（pool1_out，降採樣後 75×16 = 1200 bytes）
  //    maxpool 順序讀寫，讀指標永遠超前寫指標，安全無衝突
  //
  //  步驟 3 — Conv2 就地運算：
  //    讀取 buffer_A[1200 ~ 2399]（pool1_out）
  //    寫入 buffer_A[0 ~ 2399]  （conv2_out，75×32 = 2400 bytes）
  //    兩區域完全錯開，無讀寫衝突
  //
  //  步驟 4 — 後端借用：
  //    gap_out    → &buffer_A[1200]  (32 bytes)
  //    dense1_out → &buffer_A[1232]  (16 bytes)
  //    Conv2 輸出（前 1200 bytes）已完成，後半段可安全借用
  // -----------------------------------------------------------------------

  // Step 1. Conv1 + ReLU：輸入 nn_input，輸出至 buffer_A
  conv1d(&nn_input[0][0], buffer_A,
         CONV1_W, CONV1_B,
         NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH,
         RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu(buffer_A, NN_INPUT_LEN * NN_CONV1_OUT_CH);

  // Step 2. MaxPool1 就地池化：
  //   輸入讀取 buffer_A 前段，輸出寫入 buffer_A 後半段（偏移 1200 bytes）
  maxpool1d((int8_t(*)[NN_CONV1_OUT_CH])buffer_A,
            (int8_t(*)[NN_CONV1_OUT_CH])(&buffer_A[1200]),
            NN_INPUT_LEN);

  // Step 3. Conv2 就地運算 + ReLU：
  //   輸入讀取後半段 (&buffer_A[1200])，輸出覆蓋前半段 (buffer_A)
  conv1d(&buffer_A[1200], buffer_A,
         CONV2_W, CONV2_B,
         NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu(buffer_A, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  // Step 4. 後端記憶體借用
  int8_t* gap_out    = &buffer_A[1200];  // 借用 32 bytes
  int8_t* dense1_out = &buffer_A[1232];  // 借用 16 bytes

  // Step 5. Global Average Pooling
  // 修正：第一個參數 cast 為 flat int8_t*，並補上 rq 參數
  global_avg_pool((int8_t*)buffer_A, gap_out,
                  NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // Step 6. Dense1 + ReLU
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);

  // Step 7. Dense2（logits，無 activation）
  dense(dense1_out, logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  unsigned long latencyUs = micros() - startUs;

  // argmax：手寫 for loop，nn_ops.h 無此函式
  int label = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[label]) label = i;
  }

  // === 新增：latency timing — 推論完成時間戳（argmax 之後）===
  unsigned long t_inference_done_us = micros();
  // ===========================================================

  // 輸出格式與 SRAM3 原始碼一致
  Serial.print("# INFERENCE_LATENCY_US,");
  Serial.println(latencyUs);
  Serial.print("# PREDICTION,");
  Serial.println(LABEL_NAMES[label]);
  Serial.print("# static_ratio: ");
  Serial.println(static_ratio);

  // === 新增：latency timing — [RESULT] 機器可解析格式 ===
  // 格式：[RESULT] trial=N first=T1 last=T2 pred=T3
  //   trial : 本次錄製序號（sampleId）
  //   first : 第一筆取樣的 micros() 時間戳
  //   last  : 最後一筆取樣的 micros() 時間戳
  //   pred  : CNN 推論完成後的 micros() 時間戳
  // analyze_log.py 會解析此行計算 end-to-end latency
  Serial.print("[RESULT] trial=");   Serial.print(sampleId);
  Serial.print(" first=");           Serial.print(t_first_sample_us);
  Serial.print(" last=");            Serial.print(t_last_sample_us);
  Serial.print(" pred=");            Serial.println(t_inference_done_us);
  // ========================================================
}
