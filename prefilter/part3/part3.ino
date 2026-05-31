#include "I2C_GPIO.h"
#include "nn_ops.h"

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
#define NUM_SAMPLES         150       // 100 Hz * 1.5 sec

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
#define STATIC_ACCEL_THRESH  0.15f   // g  [修改: 0.08 → 0.15，涵蓋手持靜止]
#define STATIC_GYRO_THRESH   25.0f   // deg/s  [修改: 15.0 → 25.0，涵蓋手持靜止]
#define STATIC_FRAMES        10      // 連續幀數（維持不變）

// [STATIC FILTER] 靜止幀比例閾值
// 整段錄製中，靜止幀比例 >= 此值 → 判定整段為 static，跳過 CNN
// [修改] 0.70 → 0.60：手持時偶發超標幀較多，放寬比例門檻更穩健
#define STATIC_RATIO_THRESH  0.60f   // [修改: 0.70 → 0.60]

// [STATIC FILTER] 全域狀態變數
static float  prev_accel_mag  = 1.0f;  // 初始化為 1g（靜止重力）
static int    static_counter  = 0;     // 連續靜止幀計數器
// -----------------------------------------------------------------------

bool isRecording = false;

unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;

uint32_t sampleId = 0;

int16_t raw_samples[NN_INPUT_LEN][NN_INPUT_CH];
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];
int8_t conv1_out[NN_INPUT_LEN][NN_CONV1_OUT_CH];
int8_t pool1_out[NN_INPUT_LEN / 2][NN_CONV1_OUT_CH];
int8_t conv2_out[NN_INPUT_LEN / 2][NN_CONV2_OUT_CH];
int8_t gap_out[NN_CONV2_OUT_CH];
int8_t dense1_out[NN_DENSE1_OUT];
int8_t logits[NN_NUM_CLASSES];

// [STATIC FILTER] 4 classes：static 不進 model，其餘三個由 model 決定
const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

void MPU6050_wakeup();
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz);
void handleButton();
void recordOneSample();
void runInference();

// [STATIC FILTER] 新增函式宣告
bool isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw);

// Part1: I2C Communication and Data Collection
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
  // Your implementation
  I2C_start();
  I2C_write_byte((MPU_ADDR << 1) | 0);  // SLA+W
  I2C_write_byte(PWR_MGMT_1);           // register address: 0x6B
  I2C_write_byte(0x00);                 // data: clear sleep bit
  I2C_stop();
}

/*******************************************************************************/
// Todo 1.2: Read raw accelerometer data from MPU6050.
//
// This function performs a complete I2C register read transaction using
// the GPIO-based I2C implementation.
//
// Requirements:
// 1. Use a register pointer write to select the starting register.
// 2. Use a repeated START condition to switch to read mode.
// 3. Read six bytes of data (XH, XL, YH, YL, ZH, ZL).
// 4. Combine high and low bytes into 16-bit signed integers.
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
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz) {
  // Your implementation

  // Step 1: register pointer write
  I2C_start();
  bool ack = I2C_write_byte((MPU_ADDR << 1) | 0);  // SLA+W
  if (!ack) { I2C_stop(); return false; }
  I2C_write_byte(ACCEL_XOUT_H);                     // register address

  // Step 2: repeated START, switch to read mode
  I2C_repeated_start();
  ack = I2C_write_byte((MPU_ADDR << 1) | 1);        // SLA+R
  if (!ack) { I2C_stop(); return false; }

  // Step 3: read 14 bytes (accel + temp + gyro)
  uint8_t buf[14];
  for (int i = 0; i < 13; i++) {
    buf[i] = I2C_read_byte(true);   // ACK
  }
  buf[13] = I2C_read_byte(false);   // NACK (last byte)

  I2C_stop();

  // Step 4: combine high and low bytes
  *ax = (int16_t)((buf[0]  << 8) | buf[1]);
  *ay = (int16_t)((buf[2]  << 8) | buf[3]);
  *az = (int16_t)((buf[4]  << 8) | buf[5]);
  // buf[6..7] = temperature, discarded
  *gx = (int16_t)((buf[8]  << 8) | buf[9]);
  *gy = (int16_t)((buf[10] << 8) | buf[11]);
  *gz = (int16_t)((buf[12] << 8) | buf[13]);

  return true;
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


void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);

  // debug：確認 MPU6050 有回應
  I2C_start();
  bool ack = I2C_write_byte((MPU_ADDR << 1) | 0);
  I2C_stop();
  Serial.print("# MPU6050 ping ACK: ");
  Serial.println(ack ? "OK" : "FAIL");

  MPU6050_wakeup();
  analogWrite(LED_PIN, 0);

  Serial.println("Dataset collector ready.");
  Serial.println("Push btn to start recording.");
}

void loop() {
  handleButton();

  // [STATIC FILTER] 在 loop() 中持續偵測靜止狀態（不依賴錄製觸發）
  // 讓 static_counter 在按鈕按下前就已累積，避免第一幀誤判
  int16_t ax, ay, az, gx, gy, gz;
  bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);
  if (ok) {
    isStatic(ax, ay, az, gx, gy, gz);  // 持續更新 static_counter
  }
}

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
      }
    }
  }

  lastButtonReading = reading;
}

void recordOneSample() {
  isRecording = true;

  Serial.println("# GET_READY");
  digitalWrite(LED_PIN, HIGH);
  delay(1000);

  Serial.println("# START");
  Serial.println("sample_id,timestep,t_us,ax,ay,az,gx,gy,gz");

  unsigned long nextSampleUs = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    while ((long)(micros() - nextSampleUs) < 0) {
      // wait until next 10 ms slot
    }

    unsigned long tUs = micros();
    // ****************************************************************************
    // TODO 2.2: Declare six int16_t variables: ax, ay, az, gx, gy, gz.
    //           (The call to readImu6_raw() and error handling below depend on
    //           these variables being declared here.)
    int16_t ax, ay, az, gx, gy, gz;

    // ****************************************************************************
    bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

    if (!ok) {
      Serial.print("# ERROR at timestep ");
      Serial.println(i);
      digitalWrite(LED_PIN, LOW);
      isRecording = false;
      return;
    }

    // Store into raw_samples buffer for NN inference
    raw_samples[i][0] = ax;
    raw_samples[i][1] = ay;
    raw_samples[i][2] = az;
    raw_samples[i][3] = gx;
    raw_samples[i][4] = gy;
    raw_samples[i][5] = gz;

    // Print CSV row
    Serial.print(sampleId);   Serial.print(',');
    Serial.print(i);          Serial.print(',');
    Serial.print(tUs);        Serial.print(',');
    Serial.print(ax);         Serial.print(',');
    Serial.print(ay);         Serial.print(',');
    Serial.print(az);         Serial.print(',');
    Serial.print(gx);         Serial.print(',');
    Serial.print(gy);         Serial.print(',');
    Serial.println(gz);

    nextSampleUs += SAMPLE_INTERVAL_US;
  }

  Serial.println("# END");
  sampleId++;

  // Run inference after recording
  runInference();

  digitalWrite(LED_PIN, LOW);
  isRecording = false;
}

void runInference() {
  // [STATIC FILTER] -------------------------------------------------------
  // 在進入 CNN 推論前，先對整段錄製的 raw_samples 做靜止判斷
  //
  // 策略：掃描全部 NUM_SAMPLES 幀，統計有多少幀被判定為靜止
  //   若靜止幀比例 >= STATIC_RATIO_THRESH（60%），直接輸出 "static"
  //   否則進入原本的 CNN inference pipeline
  //
  // 為何用比例而非單一幀判斷：
  //   錄製過程中，按下按鈕的瞬間可能有輕微抖動，
  //   用比例可容忍少量非靜止幀，更穩健
  //
  // 重置全域狀態，讓每次 runInference() 都是獨立的靜止判斷
  // -----------------------------------------------------------------------
  static_counter = 0;
  prev_accel_mag = 1.0f;  // 重置為靜止重力初始值
  int static_frame_count = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    if (isStatic(raw_samples[i][0], raw_samples[i][1], raw_samples[i][2],
                 raw_samples[i][3], raw_samples[i][4], raw_samples[i][5])) {
      static_frame_count++;
    }
  }

  float static_ratio = (float)static_frame_count / (float)NUM_SAMPLES;

  // [STATIC FILTER] 靜止判斷：超過閾值比例直接輸出，不進 CNN
  if (static_ratio >= STATIC_RATIO_THRESH) {
    Serial.println("# INFERENCE_RESULT: static");
    Serial.print("# static_ratio: ");
    Serial.println(static_ratio);
    return;  // 跳過 CNN inference，節省計算資源
  }

  // -----------------------------------------------------------------------
  // 以下為原本的 CNN inference pipeline（靜止判斷未觸發才執行）
  // -----------------------------------------------------------------------

  // Preprocess: raw int16 → quantized int8 NN inputs
  preprocess_input(raw_samples, nn_input);

  // Conv1 + ReLU
  // conv1d 接受 flat int8_t* 指標，直接 cast 2D array 首元素位址
  conv1d((int8_t*)nn_input, (int8_t*)conv1_out,
         CONV1_W, CONV1_B,
         NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH,
         RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu((int8_t*)conv1_out, NN_INPUT_LEN * NN_CONV1_OUT_CH);

  // MaxPool1
  // maxpool1d 簽名：(const int8_t[][NN_CONV1_OUT_CH], int8_t[][NN_CONV1_OUT_CH], int len)
  // 直接傳 2D array，不需要 cast，也不接受 rq 參數（pool 不做 requantize）
  maxpool1d(conv1_out, pool1_out, NN_INPUT_LEN);

  // Conv2 + ReLU
  // pool1_out 是 [NN_INPUT_LEN/2][NN_CONV1_OUT_CH]，flatten 後傳入 conv1d
  conv1d((int8_t*)pool1_out, (int8_t*)conv2_out,
         CONV2_W, CONV2_B,
         NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu((int8_t*)conv2_out, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  // Global Average Pooling
  // 正確函式名稱為 global_avg_pool（不是 gap1d）
  // 簽名：(const int8_t* input, int8_t* output, int len, int ch, int32_t rq_mult, uint8_t rq_shift)
  global_avg_pool((int8_t*)conv2_out, gap_out,
                  NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // Dense1 + ReLU
  // 簽名：(const int8_t* input, int8_t* output, const int8_t weights[],
  //        const int32_t bias[], int in_features, int out_features,
  //        int32_t rq_mult, uint8_t rq_shift)
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);

  // Dense2 (logits, no activation)
  dense(dense1_out, logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  // Argmax → predicted class index
  int pred = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[pred]) pred = i;
  }

  Serial.print("# INFERENCE_RESULT: ");
  Serial.println(LABEL_NAMES[pred]);
  Serial.print("# static_ratio: ");
  Serial.println(static_ratio);
}
