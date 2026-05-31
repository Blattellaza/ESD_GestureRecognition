#ifndef NN_OPS_H
#define NN_OPS_H

#include <stdint.h>
#include "nn_weights.h"



// TODO 2.1: Implement preprocess_input() — convert raw int16 IMU samples to quantized int8 NN inputs.
//
// The process for each sample (i) and channel (c):
//   1. PHYSICAL: convert raw ADC counts to physical units
//        accel (c=0,1,2): physical = raw / 16384.0f   → units: g
//        gyro  (c=3,4,5): physical = raw / 131.0f     → units: deg/s
//   2. STANDARDIZE: zero-mean, unit-variance normalization
//        standardized = (physical - NN_MEAN[c]) / NN_STD[c]
//   3. QUANTIZE: map standardized float → int8
//        q = round(standardized / SCALE_INPUT)
//        clamp q to [-128, 127], then store as int8.
inline void preprocess_input(
    const int16_t raw[NN_INPUT_LEN][NN_INPUT_CH],
    int8_t out[NN_INPUT_LEN][NN_INPUT_CH]
) {
  for (int i = 0; i < NN_INPUT_LEN; i++) {
    for (int c = 0; c < NN_INPUT_CH; c++) {
      // Your implementation

      // Step 1: PHYSICAL — convert ADC counts to physical units
      float physical;
      if (c < 3) {
        physical = (float)raw[i][c] / 16384.0f;  // accel: g
      } else {
        physical = (float)raw[i][c] / 131.0f;    // gyro: deg/s
      }

      // Step 2: STANDARDIZE — zero-mean, unit-variance
      float standardized = (physical - NN_MEAN[c]) / NN_STD[c];

      // Step 3: QUANTIZE — map to int8
      float fq = standardized / SCALE_INPUT;
      int32_t q = (int32_t)roundf(fq);

      // Clamp to [-128, 127]
      if (q >  127) q =  127;
      if (q < -128) q = -128;

      out[i][c] = (int8_t)q;
    }
  }
}



// --------------------------------------------------------
// TODO 3.2: Requantize acc (INT32) → INT8 using fixed-point scale.
// Real value = Scale × Quantized value
// Scale = mult / 2^shift  (e.g. CONV1: 3114499 / 2^30 ≈ 0.0029)
//
// 1. SCALE: Cast 'acc' to int64_t, then multiply by 'mult'.
//           (int64_t is required: int32 × int32 can exceed 32-bit range)
// 2. SHIFT: Arithmetic right-shift the int64_t result by 'shift' using >>.
//           Divides by 2^shift, approximating the scale factor.
// 3. CLAMP: Constrain the int32_t result to [-128, 127] to prevent
//           wrap-around errors when casting to int8_t.
// 4. CAST:  Return the final int8_t value.
// --------------------------------------------------------
inline int8_t requantize_int8(int32_t acc, int32_t mult, uint8_t shift) {
  // Your implementation

  // Step 1: SCALE — 升到 int64 避免 int32 × int32 溢位
  int64_t scaled = (int64_t)acc * (int64_t)mult;

  // Step 2: SHIFT — 算術右移除以 2^shift
  int32_t result = (int32_t)(scaled >> shift);

  // Step 3: CLAMP — 限制在 int8 範圍
  if (result >  127) result =  127;
  if (result < -128) result = -128;

  // Step 4: CAST
  return (int8_t)result;
}

inline int32_t round_div(int32_t x, int32_t d) {
  if (d <= 0) return 0;
  int32_t offset = d / 2;
  if (x >= 0) return (x + offset) / d;
  return -(((-x) + offset) / d);
}


inline void relu(int8_t *data, int len) {
  for (int i = 0; i < len; i++) {
    if (data[i] < 0) data[i] = 0;
  }
}

//  PART 3: Buffer overflow prevention in MAC operations
inline void conv1d(const int8_t *input, int8_t *output, const int8_t weights[], const int32_t bias[], int len, int in_ch, int out_ch, int32_t rq_mult, uint8_t rq_shift) {
  const int kernel = 3;
  const int pad = 1;
  for (int i = 0; i < len; i++) {
    for (int oc = 0; oc < out_ch; oc++) {
      int32_t sum = bias[oc];
      for (int k = 0; k < kernel; k++) {
        int idx = i + k - pad;
        if (idx < 0 || idx >= len) continue;
        int base = oc * kernel * in_ch + k * in_ch;
        for (int ic = 0; ic < in_ch; ic++) {
          // TODO 3.1: sum += (data type)weights[base + ic] * (data type)input[idx * in_ch + ic];
          // Hint: weights and inputs are int8_t, what data type should you use for the product to prevent overflow? What about the sum?
          /* Your code here */

          // int8 × int8 最大可達 127*127 = 16129，雖然不會超過 int16，
          // 但累加多次後 int16 會溢位，因此乘積先轉成 int32 再加到 int32 sum。
          sum += (int32_t)weights[base + ic] * (int32_t)input[idx * in_ch + ic];
        }
      }
      output[i * out_ch + oc] = requantize_int8(sum, rq_mult, rq_shift);
    }
  }
}

inline void maxpool1d(const int8_t input[][NN_CONV1_OUT_CH], int8_t output[][NN_CONV1_OUT_CH], int len) {
  int out_len = len / 2;
  for (int i = 0; i < out_len; i++) {
    for (int c = 0; c < NN_CONV1_OUT_CH; c++) {
      int8_t a = input[i * 2][c];
      int8_t b = input[i * 2 + 1][c];
      output[i][c] = (a > b) ? a : b;
    }
  }
}

inline void global_avg_pool(const int8_t *input, int8_t *output, int len, int ch, int32_t rq_mult, uint8_t rq_shift) {
  for (int c = 0; c < ch; c++) {
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
      sum += (int32_t)input[i * ch + c];
    }
    // Average: divide by len using round_div, then requantize
    int32_t avg = round_div(sum, len);
    output[c] = requantize_int8(avg, rq_mult, rq_shift);
  }
}

inline void dense(const int8_t *input, int8_t *output, const int8_t weights[], const int32_t bias[], int in_features, int out_features, int32_t rq_mult, uint8_t rq_shift) {
  for (int o = 0; o < out_features; o++) {
    int32_t sum = bias[o];
    for (int i = 0; i < in_features; i++) {
      // TODO 3.1: same overflow-safe MAC as conv1d
      sum += (int32_t)weights[o * in_features + i] * (int32_t)input[i];
    }
    output[o] = requantize_int8(sum, rq_mult, rq_shift);
  }
}

#endif // NN_OPS_H
