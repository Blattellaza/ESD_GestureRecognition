#include "I2C_GPIO.h"
#include "nn_ops.h"
#include <FreeRTOS.h>


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

const char *LABEL_NAMES[NN_NUM_CLASSES] = {
  "circle",
  "left_right",
  "updown",
  "static"
};

void MPU6050_wakeup();
bool readImu6_raw(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz);
void handleButton();
void recordOneSample();
void runInference();

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

void loop() {
  handleButton();
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
    // ****************************************************************************
    // TODO 2.3: Copy the six axis values into raw_samples[i][].
    //           Channel mapping: 0→ax, 1→ay, 2→az (accelerometer, ±2 g range)
    //                            3→gx, 4→gy, 5→gz (gyroscope, ±250 °/s range)
    //           raw_samples holds int16_t ADC counts; preprocess_input() called
    //           later in runInference() converts them to physical units.
    raw_samples[i][0] = ax;
    raw_samples[i][1] = ay;
    raw_samples[i][2] = az;
    raw_samples[i][3] = gx;
    raw_samples[i][4] = gy;
    raw_samples[i][5] = gz;

    // ****************************************************************************
    // TODO 2.4: Print one CSV line per sample in the format:
    //           sampleId,i,tUs,ax,ay,az,gx,gy,gz
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

  // Run inference after recording
  runInference();

  sampleId++;
  digitalWrite(LED_PIN, LOW);
  isRecording = false;
}

void runInference() {
  // ============================================================
  // Latency measurement: from readImu6_raw() to last dense()
  // ============================================================
  int16_t ax, ay, az, gx, gy, gz;

  unsigned long t_start = micros();   // ← 計時起點：readImu6_raw() 呼叫前

  readImu6_raw(&ax, &ay, &az, &gx, &gy, &gz);

  // ── Preprocess ──────────────────────────────────────────────
  preprocess_input(raw_samples, nn_input);

  // ── Conv1 + ReLU + MaxPool ───────────────────────────────────
  conv1d((const int8_t *)nn_input, (int8_t *)conv1_out,
         CONV1_W, CONV1_B,
         NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH,
         RQ_MULT_CONV1, RQ_SHIFT_CONV1);

  relu((int8_t *)conv1_out, NN_INPUT_LEN * NN_CONV1_OUT_CH);

  maxpool1d(conv1_out, pool1_out, NN_INPUT_LEN);

  // ── Conv2 + ReLU ─────────────────────────────────────────────
  conv1d((const int8_t *)pool1_out, (int8_t *)conv2_out,
         CONV2_W, CONV2_B,
         NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH,
         RQ_MULT_CONV2, RQ_SHIFT_CONV2);

  relu((int8_t *)conv2_out, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  // ── Global Average Pool ──────────────────────────────────────
  global_avg_pool((const int8_t *)conv2_out, gap_out,
                  NN_INPUT_LEN / 2, NN_CONV2_OUT_CH,
                  RQ_MULT_GAP, RQ_SHIFT_GAP);

  // ── Dense1 + ReLU ────────────────────────────────────────────
  dense(gap_out, dense1_out,
        DENSE1_W, DENSE1_B,
        NN_CONV2_OUT_CH, NN_DENSE1_OUT,
        RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);

  relu(dense1_out, NN_DENSE1_OUT);

  // ── Dense2 (last dense) ──────────────────────────────────────
  dense(dense1_out, logits,
        DENSE2_W, DENSE2_B,
        NN_DENSE1_OUT, NN_NUM_CLASSES,
        RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);

  unsigned long t_end = micros();     // ← 計時終點：最後一個 dense() 結束後

  // ── Find argmax ──────────────────────────────────────────────
  int pred = 0;
  for (int i = 1; i < NN_NUM_CLASSES; i++) {
    if (logits[i] > logits[pred]) pred = i;
  }

  // ── Print results ────────────────────────────────────────────
  Serial.print("# t_start (us): ");
  Serial.println(t_start);

  Serial.print("# t_end   (us): ");
  Serial.println(t_end);

  Serial.print("# INFERENCE_LATENCY_US,");
  Serial.println(t_end - t_start);

  Serial.print("# PREDICTION,");
  Serial.println(LABEL_NAMES[pred]);

  Serial.println("# END");
}
