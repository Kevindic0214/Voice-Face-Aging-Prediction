#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究背景：
- 受試者會在固定1分鐘內觀看圖卡，並說出繁體中文動物名稱
- 錄音內容為受試者的即時中文語言反應
- 分析目標：從語速、停頓、詞彙重複等特徵鑑別認知功能狀態
"""

# ----------------------------------------------------------------------
# 輸入文件格式說明
# ----------------------------------------------------------------------
# 1. .txt 文件：
#    - 純文字，無多餘標記（例如：無標題、無分段）。
#    - 只有一行內容，例如："貓 狗 鳥 貓 魚 狗"
#
# 2. .srt 文件：
#    - 標準字幕格式，範例如下：
#      1
#      00:00:01,000 --> 00:00:03,500
#      貓
#      
#      2
#      00:00:04,000 --> 00:00:06,000
#      狗
#
# 3. animal_list.txt：
#    - 一行一個動物名稱，可以包含詞頻和詞性（選填）。
#    - 範例格式：
#      貓 10 n       (完整格式：詞彙 詞頻 詞性)
#      狗 8 n        (完整格式)
#      鳥            (僅有詞彙)
#      魚 6          (詞彙和詞頻，無詞性)

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import jieba

# ----------------------------------------------------------------------
# 基本設定
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,                  # ← 想看更少請改 INFO
    format="[%(levelname)s] %(message)s"
)

WINDOW_SEC = 59.0                # 錄音長度（固定為59秒）
PAUSE_TH = 2.0                   # 停頓門檻
MATTR_WINDOW: int = 20            # MATTR 滑動視窗大小

# 停頓加權設定
PAUSE_WEIGHT = {
    "level1": {"min": 2.0, "max": 5.0, "weight": 1},    # 2-5秒：權重1
    "level2": {"min": 5.0, "max": 10.0, "weight": 3},   # 5-10秒：權重3
    "level3": {"min": 10.0, "max": float('inf'), "weight": 5},  # >10秒：權重5
}

STOPWORDS = {
    # 功能詞
    "的", "了", "在", "是", "有", "和", "跟", "還有", "以及",
    "這個", "那個", "這些", "那些", "一個", "一些", "兩個", "幾個",
    "每個", "每一個", "所有", "任何", "任何人", "誰", "什麼", "哪個",
    "哪裡", "怎麼", "怎樣", "為什麼", "怎麼樣", "為什麼要", "怎麼會",
    "也", "還", "而且", "不過", "但是", "雖然", "很多", "有些", "繼續",
    # 口語填充
    "嗯", "呃", "欸", "啊", "哇", "嘛", "吧", "呢",
    "然後", "就是", "嘿", "哈", "嗯哼", "好像", "對啊", "對不對",
    "對吧", "其實", "真的", "其實是", "就是說", "所以說", "所以",
    "然後呢", "那個", "那樣", "這樣", "這麼", "這樣子", "那麼"
}
PUNCTS = set("，。！？：；、「」『』、《》〈〉（）…～-—,.!?;:\"'()[]{}")

# ----------------------------------------------------------------------
# 解析 SRT 用的正規表示式
# ----------------------------------------------------------------------
_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")


def ts2sec(ts: str) -> float:
    """HH:MM:SS,mmm → 秒(float)"""
    m = _TIME_RE.match(ts)
    if not m:
        raise ValueError(f"錯誤時間戳: {ts}")
    h, m_, s, ms = map(int, m.groups())
    return h * 3600 + m_ * 60 + s + ms / 1000.0


# ----------------------------------------------------------------------
# 只取 (start, end) 時間
# ----------------------------------------------------------------------
def parse_srt(path: Path) -> List[Tuple[float, float]]:
    """解析標準 SRT 字幕文件，提取時間戳 (start, end)。
    
    文件格式：
    - 序號
    - 時間戳（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）
    - 文字內容
    - 空白行分隔
    """
    entries: List[Tuple[float, float]] = []
    with path.open(encoding="utf-8") as fh:
        lines = (l.rstrip("\n") for l in fh)
        mode = 0
        for line in lines:
            if mode == 0 and line.isdigit():
                mode = 1
                continue
            if mode == 1 and "-->" in line:
                t1, t2 = [p.strip() for p in line.split("-->")]
                entries.append((ts2sec(t1), ts2sec(t2)))
                mode = 2
                continue
            if mode == 2 and line == "":
                mode = 0
    logging.debug(f"{path.name}: 共解析 {len(entries)} 段字幕")
    return entries


# ----------------------------------------------------------------------
# 讀取 ASR 文字
# ----------------------------------------------------------------------
def read_txt(path: Path) -> str:
    """讀取純文字文件，內容為一行且無多餘標記。
    
    範例：
    - 輸入文件內容：「貓 狗 鳥 貓 魚 狗」
    - 輸出：「貓 狗 鳥 貓 魚 狗」
    """
    txt = path.read_text(encoding="utf-8").replace("\u3000", " ").strip()
    logging.debug(f"{path.name}: 文字長度 {len(txt)} 字元")
    return txt


# ----------------------------------------------------------------------
# 斷詞
# ----------------------------------------------------------------------
def tokenize(text: str, keep_stopwords: bool) -> List[str]:
    # 先進行分詞
    tokens = jieba.lcut(text, cut_all=False, HMM=True)
    # 過濾標點和空白
    tokens = [w for w in tokens if w not in PUNCTS and w.strip()]
    # 根據需要過濾停用詞
    if not keep_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS]
    logging.debug(f"斷詞後長度 {len(tokens)}")
    return tokens


# ----------------------------------------------------------------------
# 動物詞典相關功能
# ----------------------------------------------------------------------
def load_animals(animal_file: Path = Path("animal_list.txt")) -> set:
    """從檔案載入動物詞典作為分詞的補充字典。
    
    檔案格式：
    - 每行一個動物名稱，可以包含詞頻和詞性（選填）。
    - 範例：
        貓 10 n       (完整格式：詞彙 詞頻 詞性)
        狗 8 n        (完整格式)
        鳥            (僅有詞彙)
        魚 6          (詞彙和詞頻，無詞性)
    """
    if not animal_file.exists():
        logging.error(f"找不到動物詞典檔案: {animal_file}")
        return set()
    
    animals = set()
    with animal_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            # 分割每行並至少取第一欄（動物名稱）
            parts = line.strip().split()
            if parts:  # 確保行不為空
                animal_name = parts[0]
                animals.add(animal_name)
    
    # 將動物名稱添加到jieba詞典中以提高分詞準確度
    for animal in animals:
        jieba.add_word(animal)
    
    logging.debug(f"已載入 {len(animals)} 種動物詞到分詞字典")
    return animals

# ----------------------------------------------------------------------
# MATTR implementation
# ----------------------------------------------------------------------

def mattr(tokens: List[str], window: int = MATTR_WINDOW) -> float:
    """Moving‑Average TTR. Returns NaN if tokens shorter than window."""
    n = len(tokens)
    if n < window:
        return float("nan")
    ttrs = [len(set(tokens[i:i + window])) / window for i in range(0, n - window + 1)]
    return float(np.mean(ttrs))

# ----------------------------------------------------------------------
# 核心計算
# ----------------------------------------------------------------------
def speaking_duration(segs: list[tuple[float, float]]) -> float:
    """所有字幕段長度總和（真實發聲時間）"""
    return sum(e - s for s, e in segs)

def calc_metrics(txt_path: Path, srt_path: Path) -> dict:
    # ---------- 語言部份 ----------
    text = read_txt(txt_path)

    # 載入動物詞典作為分詞補充字典
    load_animals()
    
    raw_tokens = tokenize(text, keep_stopwords=True)
    filt_tokens = tokenize(text, keep_stopwords=False)

    # ---------- 停頓部份 ----------
    segs = parse_srt(srt_path)
    if not segs:
        logging.warning(f"{srt_path.name}: 找不到字幕段，停頓數將為 0")
        segs = [(0.0, 0.0)]

    actual_speaking_time = speaking_duration(segs)

    # 計算所有停頓（段間、開頭、結尾）
    pauses: list[float] = []
    
    # 1. 開頭沉默時間（從錄音開始到第一段說話）
    first_gap = segs[0][0] - 0.0
    if first_gap > PAUSE_TH:
        pauses.append(first_gap)
    
    # 2. 段間停頓
    internal_gaps = [
        segs[i + 1][0] - segs[i][1]            # 段與段之間
        for i in range(len(segs) - 1)
    ]
    pauses.extend([g for g in internal_gaps if g > PAUSE_TH])
    
    # 3. 結尾沉默時間（從最後一段說話到錄音結束）
    last_gap = WINDOW_SEC - segs[-1][1]
    if last_gap > PAUSE_TH:
        pauses.append(last_gap)
    
    # 總停頓與總說話時間
    total_pause_time = sum(pauses)
    total_time = actual_speaking_time + total_pause_time

    # 計算語速
    sr = len(filt_tokens) / max(total_time, 1.0)

    # 計算詞彙多樣性指標
    ld_mattr = mattr(filt_tokens, window=MATTR_WINDOW)

    # 計算詞彙重複率
    word_counts = Counter(filt_tokens)
    repetition_score = sum(cnt > 1 for cnt in word_counts.values()) / len(word_counts) if word_counts else 0

    # 加權計算停頓影響
    vf_weighted = 0
    vf_counts = {"level1": 0, "level2": 0, "level3": 0}
    
    for pause in pauses:
        if PAUSE_WEIGHT["level1"]["min"] <= pause < PAUSE_WEIGHT["level1"]["max"]:
            vf_weighted += PAUSE_WEIGHT["level1"]["weight"]
            vf_counts["level1"] += 1
        elif PAUSE_WEIGHT["level2"]["min"] <= pause < PAUSE_WEIGHT["level2"]["max"]:
            vf_weighted += PAUSE_WEIGHT["level2"]["weight"]
            vf_counts["level2"] += 1
        elif pause >= PAUSE_WEIGHT["level3"]["min"]:
            vf_weighted += PAUSE_WEIGHT["level3"]["weight"]
            vf_counts["level3"] += 1

    # 計算前後半段語速差異
    half_idx = len(raw_tokens) // 2
    first_half_tokens = raw_tokens[:half_idx]
    second_half_tokens = raw_tokens[half_idx:]
    
    # 假設停頓均勻分布於前後半段
    half_time = total_time / 2
    
    sr_first_half = len(first_half_tokens) / max(half_time, 0.1)  # 避免除以0
    sr_second_half = len(second_half_tokens) / max(half_time, 0.1)
    
    # 語速比（>1表示後半段變慢）
    speech_rate_ratio = sr_first_half / max(sr_second_half, 0.1)

    # 語言流暢度 (verbal fluency)
    vf = len(filt_tokens)

    return {
        "speech_rate": round(sr, 4),
        "speech_rate_ratio": round(speech_rate_ratio, 2),
        "verbal_fluency": vf,
        f"lexical_diversity_mattr{MATTR_WINDOW}": round(ld_mattr, 4) if not np.isnan(ld_mattr) else np.nan,
        "word_repetition_score": round(repetition_score, 4),
        "voice_fluency_pauses": len(pauses),
        "voice_fluency_weighted": vf_weighted,
        "pause_level1_count": vf_counts["level1"],
        "pause_level2_count": vf_counts["level2"],
        "pause_level3_count": vf_counts["level3"],
        "total_pause_time": round(total_pause_time, 2),
        "total_speaking_time": round(actual_speaking_time, 2),
    }


# ----------------------------------------------------------------------
# 手動指定受試者 ID
# ----------------------------------------------------------------------
# SUBJECT_ID = "0005"
# TXT_DIR = Path("data/txt")
# SRT_DIR = Path("data/srt")

# txt_file = TXT_DIR / f"{SUBJECT_ID}.txt"
# srt_file = SRT_DIR / f"{SUBJECT_ID}.srt"

# if not txt_file.exists():
#     logging.error(f"{txt_file} 不存在！")
# elif not srt_file.exists():
#     logging.error(f"{srt_file} 不存在！")
# else:
#     scores = calc_metrics(txt_file, srt_file)
#     print(json.dumps(scores, ensure_ascii=False, indent=2))

# ----------------------------------------------------------------------
# Batch processing utilities
# ----------------------------------------------------------------------
def batch_process(txt_dir: Path, srt_dir: Path, out_csv: Path):
    """掃描 txt_dir 下所有 *.txt，批次計算指標並存 CSV"""
    records = []
    for txt_path in sorted(txt_dir.glob("*.txt")):
        sid = txt_path.stem          # 取檔名 (不含副檔名) 當 SUBJECT_ID
        srt_path = srt_dir / f"{sid}.srt"
        if not srt_path.exists():
            logging.warning(f"{sid}: 找不到對應的 {srt_path.name}，跳過")
            continue
        scores = calc_metrics(txt_path, srt_path)
        scores["subject_id"] = sid
        records.append(scores)
        logging.debug(f"{sid}: 完成計算 → {scores}")

    if not records:
        logging.error("整個資料夾沒有任何有效檔案！")
        return

    df = pd.DataFrame(records).set_index("subject_id")
    df.to_csv(out_csv, encoding="utf-8-sig")
    logging.info(f"已輸出彙整結果 → {out_csv}")
    print("\n============== 批次結果 ==============")
    print(df)

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TXT_DIR = Path("data/txt")
    SRT_DIR = Path("data/srt")
    OUT_CSV = Path("batch_metrics.csv")

    batch_process(TXT_DIR, SRT_DIR, OUT_CSV)
