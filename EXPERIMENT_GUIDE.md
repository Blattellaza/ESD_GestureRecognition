# ESD 手勢辨識實驗操作手冊

> **使用對象**：本手冊假設你完全不懂程式，只需照步驟執行即可。
> 如遇任何問題，把 Serial Monitor 的訊息截圖給助教即可。

---

## 目錄

1. [硬體接線](#1-硬體接線)
2. [實驗一：RTOS vs. prefilter+SRAM3 延遲比較](#2-實驗一rtos-vs-prefiltersram3-延遲比較)
3. [實驗二：Moving Average Filter 三種 window 測試](#3-實驗二moving-average-filter-三種-window-測試)
4. [如何讀 Serial 輸出](#4-如何讀-serial-輸出)
5. [如何用 analyze_log.py 分析](#5-如何用-analyze_logpy-分析)
6. [常見問題](#6-常見問題)
7. [程式碼說明（應付老師 Q&A）](#7-程式碼說明應付老師-qa)

---

## 1. 硬體接線

| Arduino Due 腳位 | 連接到           |
|-----------------|-----------------|
| Digital 8 (SCL) | MPU6050 SCL     |
| Digital 9 (SDA) | MPU6050 SDA     |
| 3.3V            | MPU6050 VCC     |
| GND             | MPU6050 GND     |
| Digital 12      | 按鈕（一端接 GND）|

> **注意**：Arduino Due 是 3.3V 系統，MPU6050 接 3.3V，不要接 5V。
> LED 已從程式碼中移除，以 Serial Monitor 文字提示代替視覺回饋。

---

## 2. 實驗一：RTOS vs. prefilter+SRAM3 延遲比較

### 2.1 燒錄 RTOS 版本

1. 打開 Arduino IDE
2. 選擇 **File → Open**，找到：
   ```
   RTOS\part3\part3.ino
   ```
3. 選擇開發板：**Tools → Board → Arduino Due (Programming Port)**
4. 選擇 COM 埠：**Tools → Port → COMx**（你的 Due 的 COM 號）
5. 點擊 **上傳**（箭頭圖示），等待「Done uploading」出現
6. 打開 Serial Monitor：**Tools → Serial Monitor**
7. 右下角 baud rate 選 **115200**
8. 你會看到：
   ```
   Dataset collector ready.
   Push btn to start recording.
   ```

### 2.2 RTOS 版本實驗步驟

1. 手持感測器（MPU6050），保持自然姿勢
2. 按下按鈕，等待 Serial Monitor 出現 `# GET_READY`（1 秒準備時間）
3. 看到 `# START` 後立刻做一個手勢（circle / left_right / updown）
4. 看到 `# END` 後代表本次完成，可以再按按鈕做下一次
5. **重複 10 次**（每個手勢各做幾次）
6. 全部選取 Serial Monitor 的內容，複製貼上到 `rtos_log.txt`

**你會看到的輸出範例**：
```
# GET_READY         ← 按鈕觸發，等 1 秒
# START             ← 現在開始做手勢！
sample_id,timestep,t_us,ax,ay,az,gx,gy,gz
0,0,12345,1024,-512,16384,10,-8,3
...（共 150 行資料）...
# [PERF] --- Sampling Period Statistics ---
# [PERF] Mean period:   10001.2 us
# [PERF] Std  period:   87.3 us
# [PERF] Max deviation: 312.0 us
# INFERENCE_RESULT: circle
[RESULT] trial=0 first=12345 last=1512345 pred=1532567
# END               ← 完成，可以再按按鈕
```

> 最重要的是 `[RESULT]` 那行，analyze_log.py 會自動解析它。

### 2.3 燒錄 prefilter+SRAM3 版本

步驟同 2.1，但開啟：
```
prefilter_with_SRAM3\part3\part3.ino
```

**重複步驟 2.2**，把輸出複製到 `prefilter_log.txt`。

---

## 3. 實驗二：Moving Average Filter 三種 window 測試

### 3.1 切換 Window Size

1. 打開：
   ```
   smoothing\part3\part3.ino
   ```
2. 找到第 **48 行**左右（搜尋 `WINDOW_SIZE`）：
   ```cpp
   #define WINDOW_SIZE 3
   ```
3. 把數字改成你要測的值（1、3 或 5），例如：
   ```cpp
   #define WINDOW_SIZE 1
   ```
4. 重新燒錄（點擊上傳）

### 3.2 Window = 1 實驗（無濾波，對照組）

1. 將 `WINDOW_SIZE` 改為 `1`，燒錄
2. 做 10 次手勢（各類各幾次）
3. 複製 Serial 輸出到 `window1_log.txt`

### 3.3 Window = 3 實驗（推薦值）

1. 將 `WINDOW_SIZE` 改為 `3`，燒錄
2. 做同樣的 10 次手勢
3. 複製到 `window3_log.txt`

### 3.4 Window = 5 實驗（強濾波）

1. 將 `WINDOW_SIZE` 改為 `5`，燒錄
2. 做同樣的 10 次手勢
3. 複製到 `window5_log.txt`

> **注意**：三次測試盡量做一樣的手勢，這樣比較有意義。

### 3.5 手動記錄 Accuracy

每次實驗結束後，在紙上記錄：
- 你做了什麼手勢
- 板子辨識出什麼（看 `# PREDICTION,xxx`）
- 對了幾次 / 共幾次

這個數字 analyze_log.py **無法自動算**，你要手動填入表格。

---

## 4. 如何讀 Serial 輸出

### 4.1 重要行說明

| 行格式 | 說明 |
|--------|------|
| `# GET_READY` | 按鈕按下，準備開始錄製 |
| `# START` | 開始取樣 |
| `[RESULT] trial=N first=T1 last=T2 pred=T3` | **最重要**：latency 計算用 |
| `[SMOOTH] window=N mean=X std=Y max_dev=Z` | 取樣穩定度（smoothing 版才有） |
| `# [PERF] Std period: X us` | 取樣間隔標準差（RTOS 版才有） |
| `# INFERENCE_RESULT: circle` | CNN 辨識出的手勢 |
| `# PREDICTION,circle` | 同上（另一格式） |
| `# INFERENCE_LATENCY_US,12345` | 純 CNN 推論耗時（微秒） |
| `Free SRAM: XXXX bytes` | 剩餘記憶體監控 |
| `# END` | 本次錄製結束 |

### 4.2 [RESULT] 行說明

```
[RESULT] trial=0 first=12345 last=1512345 pred=1532567
```

| 欄位 | 意義 |
|------|------|
| `trial` | 第幾次錄製（從 0 開始） |
| `first` | 第 1 筆感測器讀值的時間（微秒） |
| `last`  | 第 150 筆感測器讀值的時間（微秒） |
| `pred`  | CNN 推論完成的時間（微秒） |

- `last - first` ≈ 1,490,000 µs（≈ 1.49 秒，接近 1.5 秒取樣時間）
- `pred - first` = end-to-end latency（越小越好）
- `pred - last`  = 推論時間（兩版本應該接近）

---

## 5. 如何用 analyze_log.py 分析

### 5.1 前置條件

確認你的電腦有安裝 Python 3。在終端機輸入：
```
python --version
```
出現 `Python 3.x.x` 即可。

### 5.2 執行方法

1. 打開「命令提示字元」或「PowerShell」
2. 切換到 repo 目錄：
   ```
   cd C:\repos\ESD_GestureRecognition
   ```
3. 執行分析：

**只分析 RTOS：**
```
python analyze_log.py rtos_log.txt
```

**比較 RTOS 與 prefilter+SRAM3：**
```
python analyze_log.py rtos_log.txt prefilter_log.txt
```

**分析三種 window：**
```
python analyze_log.py window1_log.txt window3_log.txt window5_log.txt
```

**一次全部：**
```
python analyze_log.py rtos_log.txt prefilter_log.txt window1_log.txt window3_log.txt window5_log.txt
```

### 5.3 輸出說明

腳本執行後會：
1. 在終端機印出各檔案的統計數字
2. 產生 `analysis_report.md`（Markdown 報告，可直接貼到報告裡）

---

## 6. 常見問題

**Q：燒錄後 Serial Monitor 沒有輸出？**
- 確認 baud rate 是 115200
- 按 Arduino Due 上的 RESET 按鈕
- 確認 COM 埠選正確（Windows 裝置管理員中找 Arduino Due）

**Q：`# ERROR: Failed to wake up MPU6050` 出現？**
- 確認接線是否鬆脫（特別是 SDA/SCL）
- 確認 MPU6050 接 3.3V，不是 5V
- 重新插拔 USB

**Q：[RESULT] 行沒出現？**
- 確認你燒錄的是修改後的版本（不是舊版本）
- 搜尋 Serial Monitor 輸出中是否有 `[RESULT]`

**Q：analyze_log.py 說「沒有找到 [RESULT] 行」？**
- 確認你複製的是整段 Serial 輸出
- 確認 .txt 檔案編碼是 UTF-8（記事本存檔時選 UTF-8）

**Q：Accuracy 要怎麼算？**
- 數 Serial Monitor 中 `# INFERENCE_RESULT:` 的正確次數
- 除以總次數（不包含 static 的那幾次）

---

## 7. 程式碼說明（應付老師 Q&A）

### 7.1 為什麼用 micros() 不用 millis()？

`millis()` 精度只有 1 ms，Arduino Due 的計時誤差最差 ±2 ms。
`micros()` 精度 1 µs，比 millis() 精確 1000 倍。
我們的取樣間隔標準差 (σ) 預期只有 50~500 µs，用 millis() 會完全淹沒真實數值。

### 7.2 RTOS 版本三個 Task 分別做什麼？

| Task | 優先權 | 負責的事 |
|------|--------|---------|
| TaskSampling | 3（最高）| 等待按鈕 → 以 100Hz 取樣 150 筆 → 送到推論 queue |
| TaskInference | 2（中）| 收到樣本 → 跑 CNN → 送結果到輸出 queue |
| TaskOutput | 1（最低）| 收到結果 → Serial 印出結果 |

優先權最高的 TaskSampling 確保取樣不被推論搶佔，避免丟幀。

### 7.3 SRAM3 省了什麼記憶體？

| 移除項目 | 節省 SRAM |
|---------|-----------|
| raw_samples[150][6] | 1800 bytes |
| conv1_out / pool1_out / conv2_out | 6048 bytes |
| **合計** | **7848 bytes** |

改用 `buffer_A[2400]` 就地運算取代以上所有緩衝區。

### 7.4 Moving Average Filter 為什麼用 circular buffer？

如果用普通陣列，每次更新都要把 150 個元素往前移動一位（O(N)）。
Circular buffer 只需更新一個指標（O(1)），節省時間且不需要複製資料。
在 100Hz 取樣頻率下，O(1) vs O(N) 的差異雖小，但 SRAM 消耗一致（固定 WINDOW_SIZE 個槽）。

### 7.5 [RESULT] 行的格式設計原因

```
[RESULT] trial=N first=T1 last=T2 pred=T3
```

設計成 `key=value` 格式，方便 Python 用正規表示式解析：
- 不依賴欄位位置，增減欄位不影響解析器
- `[RESULT]` 前綴讓腳本快速過濾其他輸出行

### 7.6 prefilter+SRAM3 版本的 E2E latency 為什麼比 RTOS 長？

prefilter+SRAM3 是**序列執行**：
```
錄製 150 筆 → CNN 推論
```
E2E = 1500ms（錄製）+ CNN 時間

RTOS 是**並行 Pipeline**：
```
TaskSampling 錄製時 → TaskInference 已在處理上一批
```
對於連續使用場景，RTOS 的吞吐量（每秒可處理幾次手勢）明顯更高。
但對於單次 E2E，RTOS 版主要的優勢在於 TaskInference 不佔用 TaskSampling 的時間。

---

*本手冊由 Claude Code 自動產生，如有疑問請詢問助教。*
