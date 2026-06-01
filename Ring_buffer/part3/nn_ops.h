#ifndef NN_OPS_H
#define NN_OPS_H

#include <stdint.h>
#include "nn_weights.h"


// --------------------------------------------------------
// preprocess_input()
// convert raw int16 IMU samples to quantized int8 NN inputs.
//
// The process for each sample (i) and channel (c):
//   1. PHYSICAL: convert raw ADC counts to physical units
//        accel (c=0,1,2): physical = raw / 16384.0f   → units: g
//        gyro  (c=3,4,5): physical = raw / 131.0f     → units: deg/s
//   2. STANDARDIZE: zero-mean, unit-variance normalization
//        standardized = (physical - NN_MEAN[c]) / NN_STD[c]
//   3. [FIX 1] Soft clamp ±4σ: 防止快速動作量化飽和
//   4. QUANTIZE: map standardized float → int8
//        q = round(standardized / SCALE_INPUT)
//        clamp q to [-128, 127], then store as int8.
// --------------------------------------------------------
inline void preprocess_input(
    const int16_t raw[NN_INPUT_LEN][NN_INPUT_CH],
    int8_t out[NN_INPUT_LEN][NN_INPUT_CH]
) {
  for (int i = 0; i < NN_INPUT_LEN; i++) {
    for (int c = 0; c < NN_INPUT_CH; c++) {

      // Step 1: PHYSICAL
      float physical;
      if (c < 3) {
        physical = (float)raw[i][c] / 16384.0f;
      } else {
        physical = (float)raw[i][c] / 131.0f;
      }

      // Step 2: STANDARDIZE
      float standardized = (physical - NN_MEAN[c]) / NN_STD[c];

      // Step 3: [FIX 1] Soft clamp ±4σ
      if (standardized >  4.0f) standardized =  4.0f;
      if (standardized < -4.0f) standardized = -4.0f;

      // Step 4: QUANTIZE
      float fq = standardized / SCALE_INPUT;
      int32_t q = (int32_t)roundf(fq);
      if (q >  127) q =  127;
      if (q < -128) q = -128;

      out[i][c] = (int8_t)q;
    }
  }
}


// --------------------------------------------------------
// requantize_int8()
// Requantize acc (INT32) → INT8 using fixed-point scale.
// --------------------------------------------------------
inline int8_t requantize_int8(int32_t acc, int32_t mult, uint8_t shift) {
  int64_t scaled = (int64_t)acc * (int64_t)mult;
  int32_t result = (int32_t)(scaled >> shift);
  if (result >  127) result =  127;
  if (result < -128) result = -128;
  return (int8_t)result;
}

inline int32_t round_div(int32_t x, int32_t d) {
  if (d <= 0) return 0;
  int32_t offset = d / 2;
  if (x >= 0) return (x + offset) / d;
  return -(((-x) + offset) / d);
}


// --------------------------------------------------------
// conv1d_step()
// [RING BUFFER] 計算 Conv1 單一時間步輸出（kernel=3），含 ReLU
// --------------------------------------------------------
inline void conv1d_step(
    const int8_t input_window[3][NN_INPUT_CH],
    int8_t out_step[NN_CONV1_OUT_CH]
) {
  for (int oc = 0; oc < NN_CONV1_OUT_CH; oc++) {
    int32_t acc = CONV1_B[oc];
    for (int k = 0; k < 3; k++) {
      for (int ic = 0; ic < NN_INPUT_CH; ic++) {
        acc += (int32_t)input_window[k][ic] *
               (int32_t)CONV1_W[oc * 3 * NN_INPUT_CH + k * NN_INPUT_CH + ic];
      }
    }
    out_step[oc] = requantize_int8(acc, RQ_MULT_CONV1, RQ_SHIFT_CONV1);
    if (out_step[oc] < 0) out_step[oc] = 0;  // ReLU
  }
}


// --------------------------------------------------------
// conv1d_pool_ringbuf()
// [RING BUFFER] Conv1 + ReLU + MaxPool1 串流計算
// ring[4][16] = 64 bytes（stack），取代 conv1_out[150][16] = 2400 bytes
// same padding：左右各補 1 個零
// --------------------------------------------------------
inline void conv1d_pool_ringbuf(
    const int8_t input[NN_INPUT_LEN][NN_INPUT_CH],
    int8_t pool1_out[NN_INPUT_LEN / 2][NN_CONV1_OUT_CH]
) {
  int8_t ring[4][NN_CONV1_OUT_CH];
  int8_t window[3][NN_INPUT_CH];

  for (int t = 0; t < NN_INPUT_LEN; t++) {
    // 建立 same padding sliding window
    for (int k = 0; k < 3; k++) {
      int src = t + k - 1;
      if (src < 0 || src >= NN_INPUT_LEN) {
        for (int c = 0; c < NN_INPUT_CH; c++) window[k][c] = 0;
      } else {
        for (int c = 0; c < NN_INPUT_CH; c++) window[k][c] = input[src][c];
      }
    }

    int slot = t & 3;
    conv1d_step(window, ring[slot]);

    // 每兩步計算一次 MaxPool
    if ((t & 1) == 1) {
      int pool_idx = t >> 1;
      int slot_a   = (t - 1) & 3;
      int slot_b   = t & 3;
      for (int c = 0; c < NN_CONV1_OUT_CH; c++) {
        int8_t a = ring[slot_a][c];
        int8_t b = ring[slot_b][c];
        pool1_out[pool_idx][c] = (a > b) ? a : b;
      }
    }
  }
}


// --------------------------------------------------------
// conv1d()
// --------------------------------------------------------
inline void conv1d(
    const int8_t *input, int8_t *output,
    const int8_t weights[], const int32_t bias[],
    int len, int in_ch, int out_ch,
    int32_t rq_mult, uint8_t rq_shift
) {
  const int kernel = 3;
  const int pad    = 1;
  for (int i = 0; i < len; i++) {
    for (int oc = 0; oc < out_ch; oc++) {
      int32_t sum = bias[oc];
      for (int k = 0; k < kernel; k++) {
        int idx = i + k - pad;
        if (idx < 0 || idx >= len) continue;
        int base = oc * kernel * in_ch + k * in_ch;
        for (int ic = 0; ic < in_ch; ic++) {
          sum += (int32_t)weights[base + ic] *
                 (int32_t)input[idx * in_ch + ic];
        }
      }
      output[i * out_ch + oc] = requantize_int8(sum, rq_mult, rq_shift);
    }
  }
}

// --------------------------------------------------------
// relu()
// --------------------------------------------------------
inline void relu(int8_t *data, int len) {
  for (int i = 0; i < len; i++) {
    if (data[i] < 0) data[i] = 0;
  }
}

// --------------------------------------------------------
// maxpool1d()
// --------------------------------------------------------
inline void maxpool1d(
    const int8_t input[][NN_CONV1_OUT_CH],
    int8_t output[][NN_CONV1_OUT_CH],
    int len
) {
  int out_len = len / 2;
  for (int i = 0; i < out_len; i++) {
    for (int c = 0; c < NN_CONV1_OUT_CH; c++) {
      int8_t a = input[i * 2][c];
      int8_t b = input[i * 2 + 1][c];
      output[i][c] = (a > b) ? a : b;
    }
  }
}

// --------------------------------------------------------
// global_avg_pool()
// --------------------------------------------------------
inline void global_avg_pool(
    const int8_t *input, int8_t *output,
    int len, int ch,
    int32_t rq_mult, uint8_t rq_shift
) {
  for (int c = 0; c < ch; c++) {
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
      sum += (int32_t)input[i * ch + c];
    }
    int32_t avg = round_div(sum, len);
    output[c] = requantize_int8(avg, rq_mult, rq_shift);
  }
}

// --------------------------------------------------------
// dense()
// --------------------------------------------------------
inline void dense(
    const int8_t *input, int8_t *output,
    const int8_t weights[], const int32_t bias[],
    int in_features, int out_features,
    int32_t rq_mult, uint8_t rq_shift
) {
  for (int o = 0; o < out_features; o++) {
    int32_t sum = bias[o];
    for (int i = 0; i < in_features; i++) {
      sum += (int32_t)weights[o * in_features + i] * (int32_t)input[i];
    }
    output[o] = requantize_int8(sum, rq_mult, rq_shift);
  }
}

#endif // NN_OPS_H
