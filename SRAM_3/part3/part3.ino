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

bool isRecording = false;

unsigned long lastDebounceMs = 0;
int lastButtonReading = HIGH;
int buttonState = HIGH;

uint32_t sampleId = 0;

// int16_t raw_samples[NN_INPUT_LEN][NN_INPUT_CH];
int8_t nn_input[NN_INPUT_LEN][NN_INPUT_CH];
#define LAYER_BUFFER_SIZE 2400
int8_t buffer_A[LAYER_BUFFER_SIZE]; // 2400 bytes
int8_t buffer_B[LAYER_BUFFER_SIZE]; // 2400 bytes
// int8_t conv1_out[NN_INPUT_LEN][NN_CONV1_OUT_CH]; // 2400 bytes
// int8_t pool1_out[NN_INPUT_LEN / 2][NN_CONV1_OUT_CH]; // 1200 bytes
// int8_t conv2_out[NN_INPUT_LEN / 2][NN_CONV2_OUT_CH]; // 2400 bytes
int8_t gap_out[NN_CONV2_OUT_CH];
int8_t dense1_out[NN_DENSE1_OUT];
int8_t logits[NN_NUM_CLASSES];

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

// 適用於 Arduino Due (SAM3X8E) 的剩餘 SRAM 檢查函式
extern "C" char* sbrk(int incr);
void getFreeRam() {
  char top;
  Serial.print("Free SRAM: ");
  Serial.print(&top - reinterpret_cast<char*>(sbrk(0)));
  Serial.println(" bytes");
  // return &top - reinterpret_cast<char*>(sbrk(0));
}


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

  if (!I2C_write_byte((MPU_ADDR << 1) | 0x00) || !I2C_write_byte(PWR_MGMT_1) || !I2C_write_byte(0x00)) {
    Serial.println("# ERROR: Failed to wake up MPU6050");
  }
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

  *ax = (int16_t)((buf[0] << 8) | buf[1]);
  *ay = (int16_t)((buf[2] << 8) | buf[3]);
  *az = (int16_t)((buf[4] << 8) | buf[5]);
  // buf[6..7]  (discarded)
  *gx = (int16_t)((buf[8] << 8) | buf[9]);
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
        // Serial.print("Free SRAM: ");
        // Serial.print(getFreeRam());
        // Serial.println(" bytes");
        getFreeRam();
        
      }
    }
  }

  lastButtonReading = reading;
}

int8_t quantize_single_axis(int16_t raw_val, int32_t s, int32_t z, uint8_t shift) {
    int32_t q = ((int32_t)raw_val * s + z) >> shift;
    
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    
    return (int8_t)q;
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
    /* Your implementation */
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
    //           later in runInference() converts them to physical units and int8.
    /* Your implementation */
    // raw_samples[i][0] = ax;
    // raw_samples[i][1] = ay;
    // raw_samples[i][2] = az;
    // raw_samples[i][3] = gx;
    // raw_samples[i][4] = gy;
    // raw_samples[i][5] = gz;

    // 修改：不存入 raw_samples，而是直接即時量化並存入 nn_input
    nn_input[i][0] = quantize_single_axis(ax, PRE_S[0], PRE_Z[0], PRE_SHIFT[0]);
    nn_input[i][1] = quantize_single_axis(ay, PRE_S[1], PRE_Z[1], PRE_SHIFT[1]);
    nn_input[i][2] = quantize_single_axis(az, PRE_S[2], PRE_Z[2], PRE_SHIFT[2]);
    nn_input[i][3] = quantize_single_axis(gx, PRE_S[3], PRE_Z[3], PRE_SHIFT[3]);
    nn_input[i][4] = quantize_single_axis(gy, PRE_S[4], PRE_Z[4], PRE_SHIFT[4]);
    nn_input[i][5] = quantize_single_axis(gz, PRE_S[5], PRE_Z[5], PRE_SHIFT[5]);
    // *****************************************************************************
    Serial.print(sampleId);
    Serial.print(",");
    Serial.print(i);
    Serial.print(",");
    Serial.print(tUs);
    Serial.print(",");
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.print(az);
    Serial.print(",");
    Serial.print(gx);
    Serial.print(",");
    Serial.print(gy);
    Serial.print(",");
    Serial.println(gz);

    nextSampleUs += SAMPLE_INTERVAL_US;
  }
  Serial.println(micros());/////////////////////


  runInference();

  Serial.println("# END");
  digitalWrite(LED_PIN, LOW);
  sampleId++;
  isRecording = false;
}

void runInference() {
  // ************************************************
  // TODO 2.4: Preprocess raw_samples into nn_input using preprocess_input() implemented in nn_ops.h.
  /* Your implementation */
  // preprocess_input(raw_samples, nn_input);
  // ************************************************
  unsigned long startUs = micros();

  conv1d(&nn_input[0][0], buffer_A, CONV1_W, CONV1_B, NN_INPUT_LEN, NN_INPUT_CH, NN_CONV1_OUT_CH, RQ_MULT_CONV1, RQ_SHIFT_CONV1);
  relu(buffer_A, NN_INPUT_LEN * NN_CONV1_OUT_CH);

  maxpool1d((int8_t(*)[NN_CONV1_OUT_CH])buffer_A, (int8_t(*)[NN_CONV1_OUT_CH])buffer_B, NN_INPUT_LEN);

  conv1d(buffer_B,buffer_A, CONV2_W, CONV2_B, NN_INPUT_LEN / 2, NN_CONV1_OUT_CH, NN_CONV2_OUT_CH, RQ_MULT_CONV2, RQ_SHIFT_CONV2);
  relu(buffer_A, (NN_INPUT_LEN / 2) * NN_CONV2_OUT_CH);

  global_avg_pool((int8_t(*)[NN_CONV2_OUT_CH])buffer_A, gap_out, NN_INPUT_LEN / 2);

  dense(gap_out, dense1_out, DENSE1_W, DENSE1_B, NN_CONV2_OUT_CH, NN_DENSE1_OUT, RQ_MULT_DENSE1, RQ_SHIFT_DENSE1);
  relu(dense1_out, NN_DENSE1_OUT);
  dense(dense1_out, logits, DENSE2_W, DENSE2_B, NN_DENSE1_OUT, NN_NUM_CLASSES, RQ_MULT_DENSE2, RQ_SHIFT_DENSE2);
  
  unsigned long latencyUs = micros() - startUs;
  int label = argmax(logits, NN_NUM_CLASSES);

  Serial.print("# INFERENCE_LATENCY_US,");
  Serial.println(latencyUs);
  Serial.print("# PREDICTION,");
  Serial.println(LABEL_NAMES[label]);
}

