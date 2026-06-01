// ============================================================
// ESD Gesture Recognition
//
// 整合功能：
//   [RING BUFFER] conv1d_pool_ringbuf() 取代 conv1_out[150][16]
//                 節省 2400 bytes 全域 SRAM
//   [STATIC]      錄製迴圈中即時累積 static_frame_count
//                 靜止比例 >= STATIC_RATIO_THRESH → 跳過 CNN
//   [FIX 1]       preprocess_input() 加 ±4σ soft clamp
//   [FIX 2]       第一次推論為 left_right 時，翻轉 gz 後二次推論
// ============================================================

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
#define SAMPLE_INTERVAL_US  10000UL
#define RECORD_SECONDS      1.5
#define NUM_SAMPLES         150

// -----------------------------------------------------------------------
// [STATIC FILTER] 閾值
// -----------------------------------------------------------------------
#define STATIC_ACCEL_THRESH  0.15f   // g，相鄰幀 accel magnitude 變化量
#define STATIC_GYRO_THRESH   25.0f   // deg/s，gyro magnitude
#define STATIC_FRAMES        10      // 需連續幾幀滿足才判定靜止
#define STATIC_RATIO_THRESH  0.60f   // 整段錄製靜止幀比例門檻

// [STATIC FILTER] 全域狀態
static float prev_accel_mag = 1.0f;
static int   static_counter = 0;

// [FIX 2] gz 翻轉二次推論門檻
#define CW_CIRCLE_MARGIN  10

// -----------------------------------------------------------------------
// 全域狀態
// -----------------------------------------------------------------------
bool isRecording = false;
unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;
uint32_t sampleId = 0;

// -----------------------------------------------------------------------
// [SRAM] 緩衝區配置
//
// 移除：conv1_out[150][16] = 2400 bytes（改用 ring buffer）
// 保留：
//   raw_samples[150][6]  = 1800 bytes（供 preprocess_input 使用）
//   nn_input[150][6]     =  900 bytes
//   pool1_out[75][16]    = 1200 bytes
//   conv2_out[75][32]    = 2400 bytes
//     gap_out    借用 conv2_out + 1200（32 bytes）
//     dense1_out 借用 conv2_out + 1232（16 bytes）
//   logits[3]            =    3 bytes
// -----------------------------------------------------------------------
int16_t raw_samples[NN_INPUT_LEN][NN_INPUT_CH];        // 1800 bytes
int8_t  nn_input[NN_INPUT_LEN][NN_INPUT_CH];           //  900 bytes
int8_t  pool1_out[NN_INPUT_LEN/2][NN_CONV1_OUT_CH];    // 1200 bytes
int8_t  conv2_out[NN_INPUT_LEN/2][NN_CONV2_OUT_CH];    // 2400 bytes
int8_t  logits[NN_NUM_CLASSES];                        //    3 bytes

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown"
};

// -----------------------------------------------------------------------
// 函式宣告
// -----------------------------------------------------------------------
void MPU6050_wakeup();
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz);
void handleButton();
void recordOneSample();
void runInference(int static_frame_count);
bool isStatic(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw);
void run_cnn_pipeline(int8_t out_logits[NN_NUM_CLASSES]);

// -----------------------------------------------------------------------
// getFreeRam()
// -----------------------------------------------------------------------
extern "C" char* sbrk(int incr);
void getFreeRam() {
  char top;
  Serial.print("# Free SRAM: ");
  Serial.print(&top - reinterpret_cast<char*>(sbrk(0)));
  Serial.println(" bytes");
}


// ============================================================
// MPU6050_wakeup()
// ============================================================
void MPU6050_wakeup() {
  I2C_start();
  I2C_write_byte((MPU_ADDR << 1) | 0);
  I2C_write_byte(PWR_MGMT_1);
  I2C_write_byte(0x00);
  I2C_stop();
}


// ============================================================
// readImu6_raw()
// ============================================================
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz) {
  I2C_start();
  bool ack = I2C_write_byte((MPU_ADDR << 1) | 0);
  if (!ack) { I2C_stop(); return false; }
  I2C_write_byte(ACCEL_XOUT_H);

  I2C_repeated_start();
  ack = I2C_write_byte((MPU_ADDR << 1) | 1);
  if (!ack) { I2C_stop(); return false; }

  uint8_t buf[14];
  for (int i = 0; i < 13; i++) buf[i] = I2C_read_byte(true);
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
// isStatic()
// 雙重條件：accel magnitude 變化量 + gyro magnitude
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
// run_cnn_pipeline()
// [RING BUFFER] Conv1+ReLU+Pool1 用 conv1d_pool_ringbuf()
// [SRAM]        gap_out / dense1_out 借用 conv2_out 後段
// ============================================================
void run_cnn_pipeline(int8_t out_logits[NN_NUM_CLASSES]) {

  // Step 1+2: Conv1 + ReLU + MaxPool1（ring buffer，64 bytes stack）
  conv1d_pool_ringbuf(nn_input, pool1_out);

  // Step 3: Conv2 + ReLU
  conv1d((const int8_t*)pool1_out, (int8_t*)conv2_out,
         CONV2_W, CONV2_B,
         NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu((int8_t*)conv2_out, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  // Step 4: 借用 conv2_out 後段存放 gap_out / dense1_out
  // conv2_out 總大小 = 75 × 32 = 2400 bytes
  // GAP 讀完 conv2_out 後，1200 bytes 以後的空間可安全借用
  int8_t* gap_out    = (int8_t*)conv2_out + 1200;  // 32 bytes
  int8_t* dense1_out = (int8_t*)conv2_out + 1232;  // 16 bytes

  // Step 5: Global Average Pooling
  global_avg_pool((const int8_t*)conv2_out, gap_out,
                  NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // Step 6: Dense1 + ReLU
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);

  // Step 7: Dense2（logits）
  dense(dense1_out, out_logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);
}


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

  // [STATIC] loop() 持續更新 prev_accel_mag 預熱
  // recordOneSample() 開頭會重置，確保錄製段獨立判斷
  int16_t ax, ay, az, gx, gy, gz;
  if (readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz)) {
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

  // [STATIC] 重置，確保每次錄製獨立判斷
  static_counter = 0;
  prev_accel_mag = 1.0f;
  int static_frame_count = 0;

  Serial.println("# GET_READY");
  digitalWrite(LED_PIN, HIGH);
  delay(1000);

  Serial.println("# START");
  Serial.println("sample_id,timestep,t_us,ax,ay,az,gx,gy,gz");

  unsigned long nextSampleUs = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    while ((long)(micros() - nextSampleUs) < 0) {}

    unsigned long tUs = micros();

    // TODO 2.2
    int16_t ax, ay, az, gx, gy, gz;

    bool ok = readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

    if (!ok) {
      Serial.print("# ERROR at timestep ");
      Serial.println(i);
      digitalWrite(LED_PIN, LOW);
      isRecording = false;
      return;
    }

    // TODO 2.3: 存入 raw_samples
    raw_samples[i][0] = ax;
    raw_samples[i][1] = ay;
    raw_samples[i][2] = az;
    raw_samples[i][3] = gx;
    raw_samples[i][4] = gy;
    raw_samples[i][5] = gz;

    // [STATIC] 即時判斷靜止，與錄製並行
    if (isStatic(ax, ay, az, gx, gy, gz)) {
      static_frame_count++;
    }

    // TODO 2.4: CSV output
    Serial.print(sampleId); Serial.print(",");
    Serial.print(i);        Serial.print(",");
    Serial.print(tUs);      Serial.print(",");
    Serial.print(ax);       Serial.print(",");
    Serial.print(ay);       Serial.print(",");
    Serial.print(az);       Serial.print(",");
    Serial.print(gx);       Serial.print(",");
    Serial.print(gy);       Serial.print(",");
    Serial.println(gz);

    nextSampleUs += SAMPLE_INTERVAL_US;
  }

  Serial.println("# DATASET_END");

  runInference(static_frame_count);

  sampleId++;
  digitalWrite(LED_PIN, LOW);
  isRecording = false;
}


// ============================================================
// runInference()
// ============================================================
void runInference(int static_frame_count) {

  // [STATIC] 計算靜止比例
  float static_ratio = (float)static_frame_count / (float)NUM_SAMPLES;

  if (static_ratio >= STATIC_RATIO_THRESH) {
    Serial.print("# INFERENCE_LATENCY_US,");
    Serial.println(0);
    Serial.println("# PREDICTION,static");
    Serial.print("# static_ratio: ");
    Serial.println(static_ratio);
    Serial.println("# END");
    return;
  }

  // [FIX 1] preprocess_input 含 ±4σ soft clamp（在 nn_ops.h 內）
  preprocess_input(raw_samples, nn_input);

  unsigned long t_start = micros();

  // [FIX 2] 第一次推論
  run_cnn_pipeline(logits);

  int label = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[label]) label = i;
  }

  // [FIX 2] 若判為 left_right，翻轉 gz 後二次推論
  if (label == 1) {
    // 翻轉 gz（channel 5）
    for (int i = 0; i < NN_INPUT_LEN; i++) {
      int16_t flipped = -(int16_t)nn_input[i][5];
      if (flipped >  127) flipped =  127;
      if (flipped < -128) flipped = -128;
      nn_input[i][5] = (int8_t)flipped;
    }

    // 第二次推論
    int8_t logits_flipped[NN_NUM_CLASSES];
    run_cnn_pipeline(logits_flipped);

    // 若翻轉後 circle(0) 超過 left_right(1) 達門檻，改判為 circle
    if (logits_flipped[0] > logits_flipped[1] + CW_CIRCLE_MARGIN) {
      label = 0;
      Serial.println("# [FIX2] CW circle detected via gz-flip");
    }

    // 還原 gz
    for (int i = 0; i < NN_INPUT_LEN; i++) {
      int16_t restored = -(int16_t)nn_input[i][5];
      if (restored >  127) restored =  127;
      if (restored < -128) restored = -128;
      nn_input[i][5] = (int8_t)restored;
    }
  }

  unsigned long t_end = micros();

  Serial.print("# t_start (us): ");
  Serial.println(t_start);
  Serial.print("# t_end   (us): ");
  Serial.println(t_end);
  Serial.print("# INFERENCE_LATENCY_US,");
  Serial.println(t_end - t_start);
  Serial.print("# PREDICTION,");
  Serial.println(LABEL_NAMES[label]);
  Serial.print("# static_ratio: ");
  Serial.println(static_ratio);
  Serial.println("# END");
}
