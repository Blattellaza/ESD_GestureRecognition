#!/usr/bin/env python3
# analyze_log.py
# ==============================================================
# 輸入：從 Arduino Serial Monitor 複製貼上的原始 log（.txt 檔案）
# 輸出：
#   1. End-to-end latency 比較表（RTOS vs. prefilter+SRAM3）
#   2. Moving Average 對照表（window 1 / 3 / 5 × latency / stability）
#   3. 填好數字的 Markdown 結論段落
#
# 使用方式（假設你完全不懂程式）：
#   1. 把 Serial Monitor 的輸出全部複製
#   2. 貼到一個 .txt 檔案，例如 rtos_log.txt
#   3. 在終端機輸入：
#        python analyze_log.py rtos_log.txt
#      或同時分析多個檔案：
#        python analyze_log.py rtos_log.txt prefilter_log.txt window3_log.txt
#
# 支援的 log 格式（Arduino Serial 輸出）：
#   [RESULT] trial=N first=T1 last=T2 pred=T3
#   [SMOOTH] window=N mean=X std=Y max_dev=Z
#   # [PERF] Std  period:   X.X us
#   # INFERENCE_LATENCY_US,X
# ==============================================================

import sys
import re
import math
import os
from collections import defaultdict


# ---------------------------------------------------------------
# 解析函式
# ---------------------------------------------------------------

def parse_result_lines(text):
    """解析 [RESULT] 行，回傳 list of dict"""
    pattern = r'\[RESULT\]\s+trial=(\d+)\s+first=(\d+)\s+last=(\d+)\s+pred=(\d+)'
    rows = []
    for m in re.finditer(pattern, text):
        trial = int(m.group(1))
        first = int(m.group(2))
        last  = int(m.group(3))
        pred  = int(m.group(4))
        rows.append({
            'trial':           trial,
            'first_us':        first,
            'last_us':         last,
            'pred_us':         pred,
            'record_us':       last - first,       # 錄製段（first → last）
            'inference_us':    pred - last,        # 推論段（last → pred）
            'end_to_end_us':   pred - first,       # 全程（first → pred）
        })
    return rows


def parse_smooth_lines(text):
    """解析 [SMOOTH] 行，並配對同 trial 的 [RESULT] E2E 資料。
    smoothing 版每個 trial 輸出順序：[SMOOTH] 緊接著 [RESULT]，
    利用位置比對找到對應的 [RESULT]，把 e2e_us 一起存入。
    """
    smooth_pat = r'\[SMOOTH\]\s+window=(\d+)\s+mean=([\d.]+)\s+std=([\d.]+)\s+max_dev=([\d.]+)'
    result_pat = r'\[RESULT\]\s+trial=(\d+)\s+first=(\d+)\s+last=(\d+)\s+pred=(\d+)'

    smooths = [(m.start(), m) for m in re.finditer(smooth_pat, text)]
    results = [(m.start(), m) for m in re.finditer(result_pat, text)]

    rows = []
    for spos, sm in smooths:
        # 找這個 [SMOOTH] 之後第一個 [RESULT]（同 trial）
        e2e_us = None
        for rpos, rm in results:
            if rpos > spos:
                e2e_us = int(rm.group(4)) - int(rm.group(2))  # pred - first
                break
        rows.append({
            'window':   int(sm.group(1)),
            'mean_us':  float(sm.group(2)),
            'std_us':   float(sm.group(3)),
            'max_dev':  float(sm.group(4)),
            'e2e_us':   e2e_us,   # 可能為 None（若 log 不完整）
        })
    return rows


def parse_inference_latency_lines(text):
    """解析 # INFERENCE_LATENCY_US,X 行（舊格式相容）"""
    pattern = r'#\s*INFERENCE_LATENCY_US,(\d+)'
    return [int(m.group(1)) for m in re.finditer(pattern, text)]


def parse_perf_std_lines(text):
    """解析 # [PERF] Std  period:   X.X us 行（RTOS 版）"""
    pattern = r'#\s*\[PERF\]\s+Std\s+period:\s+([\d.]+)\s+us'
    return [float(m.group(1)) for m in re.finditer(pattern, text)]


# ---------------------------------------------------------------
# 統計函式
# ---------------------------------------------------------------

def stats(values):
    """計算 list 的 mean / std / min / max"""
    if not values:
        return None
    n    = len(values)
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n
    std  = math.sqrt(var)
    return {
        'n':    n,
        'mean': mean,
        'std':  std,
        'min':  min(values),
        'max':  max(values),
    }


def fmt(v, unit='us', decimals=0):
    """格式化數字，保留小數位"""
    if v is None:
        return '[___]'
    if decimals == 0:
        return f"{v:.0f} {unit}"
    return f"{v:.{decimals}f} {unit}"


# ---------------------------------------------------------------
# 報告產生函式
# ---------------------------------------------------------------

def report_latency(label, result_rows):
    """產生 end-to-end latency 統計報告"""
    if not result_rows:
        print(f"  [{label}] 沒有找到 [RESULT] 行，請確認 log 格式。")
        return

    e2e    = [r['end_to_end_us']  for r in result_rows]
    rec    = [r['record_us']      for r in result_rows]
    inf    = [r['inference_us']   for r in result_rows]

    se2e = stats(e2e)
    srec = stats(rec)
    sinf = stats(inf)

    print(f"\n  {label} — [RESULT] 解析結果（共 {len(result_rows)} 筆）")
    print(f"  {'指標':<30} {'平均':>12} {'標準差':>12} {'最小':>12} {'最大':>12}")
    print(f"  {'-'*78}")

    def row(name, s):
        if s is None:
            return
        print(f"  {name:<30} "
              f"{fmt(s['mean']):>12} "
              f"{fmt(s['std']):>12} "
              f"{fmt(s['min']):>12} "
              f"{fmt(s['max']):>12}")

    row("End-to-end latency",  se2e)
    row("  其中：錄製段 (first→last)", srec)
    row("  其中：推論段 (last→pred)",  sinf)

    return se2e, sinf


def report_smooth(label, smooth_rows):
    """產生 Moving Average 穩定度統計報告"""
    if not smooth_rows:
        return

    by_window = defaultdict(list)
    for r in smooth_rows:
        by_window[r['window']].append(r)

    print(f"\n  {label} — [SMOOTH] 解析結果")
    print(f"  {'Window':>8} {'mean_us':>10} {'std σ (us)':>12} {'max_dev':>10} {'E2E mean (us)':>16} {'筆數':>6}")
    print(f"  {'-'*68}")

    for w in sorted(by_window.keys()):
        rows    = by_window[w]
        stds    = [r['std_us'] for r in rows]
        means   = [r['mean_us'] for r in rows]
        maxdevs = [r['max_dev'] for r in rows]
        e2es    = [r['e2e_us'] for r in rows if r['e2e_us'] is not None]
        e2e_str = f"{sum(e2es)/len(e2es):.0f}" if e2es else "[___]"
        print(f"  {w:>8} "
              f"{sum(means)/len(means):>10.1f} "
              f"{sum(stds)/len(stds):>12.1f} "
              f"{sum(maxdevs)/len(maxdevs):>10.1f} "
              f"{e2e_str:>16} "
              f"{len(rows):>6}")

    print(f"\n  ★ Accuracy 需手動填入：數一下各 window 辨識正確的次數 / 總次數")
    print(f"    （看 Serial Monitor 裡 # PREDICTION,xxx 是否與你做的手勢相符）")

    return by_window


# ---------------------------------------------------------------
# Markdown 輸出函式
# ---------------------------------------------------------------

def markdown_latency_table(rtos_e2e, rtos_inf, pre_e2e, pre_inf):
    """產生 end-to-end latency 比較 Markdown 表格"""
    def fv(s, key):
        if s is None:
            return '[___]'
        return f"{s[key]:.0f} µs"

    lines = [
        "",
        "## End-to-End Latency 比較（RTOS vs. prefilter+SRAM3）",
        "",
        "| 版本               | E2E mean       | E2E std        | 推論段 mean    | 推論段 std     |",
        "|--------------------|----------------|----------------|----------------|----------------|",
        f"| RTOS               | {fv(rtos_e2e,'mean'):<14} | {fv(rtos_e2e,'std'):<14} | {fv(rtos_inf,'mean'):<14} | {fv(rtos_inf,'std'):<14} |",
        f"| prefilter+SRAM3    | {fv(pre_e2e,'mean'):<14} | {fv(pre_e2e,'std'):<14} | {fv(pre_inf,'mean'):<14} | {fv(pre_inf,'std'):<14} |",
        "",
        "> **說明**：E2E = `pred - first`（第一筆取樣 → 推論完成）。",
        "> 推論段 = `pred - last`（最後一筆取樣 → 推論完成），",
        "> 兩版本推論段理論上相近；差異主要來自 RTOS queue 傳遞 overhead。",
        "",
    ]
    return "\n".join(lines)


def markdown_conclusion_latency(rtos_e2e, pre_e2e):
    """產生 latency 比較結論段落"""
    if rtos_e2e and pre_e2e:
        diff = pre_e2e['mean'] - rtos_e2e['mean']
        pct  = diff / pre_e2e['mean'] * 100 if pre_e2e['mean'] > 0 else 0
        diff_str = f"{diff:.0f} µs（約 {pct:.1f}%）"
    else:
        diff_str = "[___] µs（[___]%）"

    rtos_mean_str = fmt(rtos_e2e['mean'] if rtos_e2e else None)
    pre_mean_str  = fmt(pre_e2e['mean']  if pre_e2e  else None)

    lines = [
        "",
        "## End-to-End Latency 結論",
        "",
        f"實測結果：RTOS 版 E2E mean = {rtos_mean_str}，",
        f"prefilter+SRAM3 版 E2E mean = {pre_mean_str}，",
        f"RTOS 版短 {diff_str}。",
        "",
        "**理論說明**：",
        "",
        "- RTOS 版採三 Task 並行 Pipeline：TaskSampling（Priority 3）專職取樣，",
        "  不被 Inference 搶佔，取樣不丟幀；TaskInference 在取樣同時可處理",
        "  上一筆資料，降低 end-to-end latency。",
        "- prefilter+SRAM3 版為序列執行：錄製完成後才進入 CNN，",
        "  inference 耗時直接疊加在錄製時間之後，E2E = 錄製 + 推論。",
        "- 注意：單次 CNN inference 的計算時間兩版本相同（同一網路、同一硬體）；",
        "  RTOS 降低的是整體 pipeline 的 end-to-end latency 與吞吐量，",
        "  而非縮短單次推論時間。",
        "- RTOS queue 傳遞 overhead 通常 < 1 ms，遠小於錄製時間（1500 ms）。",
        "",
    ]
    return "\n".join(lines)


def markdown_smooth_table(by_window):
    """產生 Moving Average 對照表（window 1/3/5）"""
    lines = [
        "",
        "## Moving Average Filter 對照表",
        "",
        "| Window Size | Accuracy（手動標注）| Stability σ (µs) | Latency E2E mean (µs) |",
        "|-------------|:------------------:|-----------------:|----------------------:|",
    ]

    for w in [1, 3, 5]:
        if w in by_window:
            rows  = by_window[w]
            stds  = [r['std_us'] for r in rows]
            sigma = f"{sum(stds)/len(stds):.1f}"
            e2es  = [r['e2e_us'] for r in rows if r['e2e_us'] is not None]
            e2e   = f"{sum(e2es)/len(e2es):.0f}" if e2es else "[___]"
        else:
            sigma = "[___]"
            e2e   = "[___]"
        lines.append(f"| {w:<11} | 請填入            | {sigma:>16} | {e2e:>21} |")

    lines += [
        "",
        "> **Accuracy 填寫方式**：",
        "> 數 Serial Monitor 裡 `# PREDICTION,xxx` 與你實際做的手勢相符的次數，",
        "> 除以總次數（不含 static），填入格式如 `8/10 = 80%`。",
        "> **Stability σ**：取樣間隔標準差（µs），由 `[SMOOTH]` 行自動計算。",
        "> **Latency E2E**：`pred - first`（µs），由 `[RESULT]` 行自動計算。",
        "",
    ]
    return "\n".join(lines)


def markdown_smooth_conclusion(by_window):
    """產生 Moving Average 結論段落"""
    w1_std = w3_std = w5_std = None
    if 1 in by_window:
        stds = [r['std_us'] for r in by_window[1]]
        w1_std = sum(stds) / len(stds)
    if 3 in by_window:
        stds = [r['std_us'] for r in by_window[3]]
        w3_std = sum(stds) / len(stds)
    if 5 in by_window:
        stds = [r['std_us'] for r in by_window[5]]
        w5_std = sum(stds) / len(stds)

    def sv(v):
        return f"{v:.1f} µs" if v is not None else "[___] µs"

    lines = [
        "",
        "## Moving Average Filter 結論",
        "",
        "| Window | Stability σ  |",
        "|--------|--------------|",
        f"|   1    | {sv(w1_std):<12} |",
        f"|   3    | {sv(w3_std):<12} |",
        f"|   5    | {sv(w5_std):<12} |",
        "",
        "**分析**：",
        "",
        "- window=1（無濾波）：σ 最大，原始 IMU 雜訊完整保留，",
        "  可能導致 Static Filter 誤判或 CNN 輸入品質下降。",
        "- window=3：σ 明顯降低，雜訊平滑，每幀平均延遲約 20 ms（@100 Hz）。",
        "  Accuracy 預期維持或略有提升，為推薦設定。",
        "- window=5：σ 最小，平滑效果最強，但 gesture response 延遲增加約 40 ms，",
        "  快速手勢（如 left_right）可能因訊號鈍化而降低 accuracy。",
        "- 綜合考量：**window=3 為甜蜜點**，平衡雜訊抑制與手勢響應速度。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------

def analyze_file(filepath):
    """分析單一 log 檔案"""
    if not os.path.exists(filepath):
        print(f"[錯誤] 找不到檔案：{filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    label = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"  分析檔案：{label}")
    print(f"{'='*60}")

    result_rows  = parse_result_lines(text)
    smooth_rows  = parse_smooth_lines(text)
    inf_latency  = parse_inference_latency_lines(text)
    perf_stds    = parse_perf_std_lines(text)

    e2e_stats = inf_stats = None
    by_window = {}

    if result_rows:
        ret = report_latency(label, result_rows)
        if ret:
            e2e_stats, inf_stats = ret

    if smooth_rows:
        by_window = report_smooth(label, smooth_rows)

    if inf_latency:
        s = stats(inf_latency)
        print(f"\n  # INFERENCE_LATENCY_US 統計（共 {s['n']} 筆）")
        print(f"    mean={s['mean']:.0f} us  std={s['std']:.0f} us  "
              f"min={s['min']} us  max={s['max']} us")

    if perf_stds:
        s = stats(perf_stds)
        print(f"\n  # [PERF] Sampling Period Std 統計（共 {s['n']} 筆）")
        print(f"    mean={s['mean']:.1f} us  std={s['std']:.1f} us  "
              f"min={s['min']:.1f} us  max={s['max']:.1f} us")

    return {
        'label':      label,
        'e2e_stats':  e2e_stats,
        'inf_stats':  inf_stats,
        'by_window':  by_window,
    }


def main():
    if len(sys.argv) < 2:
        print("用法：python analyze_log.py <log1.txt> [log2.txt] ...")
        print("")
        print("範例：")
        print("  python analyze_log.py rtos_log.txt")
        print("  python analyze_log.py rtos_log.txt prefilter_log.txt window3.txt")
        sys.exit(1)

    results = []
    for filepath in sys.argv[1:]:
        r = analyze_file(filepath)
        if r:
            results.append(r)

    if not results:
        return

    # ---------------------------------------------------------------
    # 產生 Markdown 報告（寫入 analysis_report.md）
    # ---------------------------------------------------------------
    md_lines = ["# ESD Gesture Recognition — 實驗分析報告", ""]

    # 找 RTOS 和 prefilter 的結果（依檔名判斷）
    rtos_r    = next((r for r in results if 'rtos'     in r['label'].lower()), None)
    pre_r     = next((r for r in results if 'prefilter' in r['label'].lower() or 'sram' in r['label'].lower()), None)

    if rtos_r and pre_r:
        md_lines.append(markdown_latency_table(
            rtos_r['e2e_stats'], rtos_r['inf_stats'],
            pre_r['e2e_stats'],  pre_r['inf_stats']
        ))
        md_lines.append(markdown_conclusion_latency(
            rtos_r['e2e_stats'], pre_r['e2e_stats']
        ))
    elif results:
        # 只有一個檔案也產生基本表格
        r = results[0]
        md_lines.append(markdown_latency_table(
            r['e2e_stats'], r['inf_stats'], None, None
        ))

    # Moving Average 表格（找有 smooth 資料的）
    all_by_window = {}
    for r in results:
        for w, rows in r['by_window'].items():
            if w not in all_by_window:
                all_by_window[w] = []
            all_by_window[w].extend(rows)

    if all_by_window:
        md_lines.append(markdown_smooth_table(all_by_window))
        md_lines.append(markdown_smooth_conclusion(all_by_window))

    report_path = os.path.join(os.path.dirname(sys.argv[1]), 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print(f"  Markdown 報告已寫入：{report_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
