import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

import jieba

# ----------------------------------------------------------------------
# 基本設定
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,                  # ← 想看更少請改 INFO
    format="[%(levelname)s] %(message)s"
)

WINDOW_SEC = 59.0                # 錄音長度
PAUSE_TH = 2.0                   # 停頓門檻

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
    "也", "還", "而且", "不過", "但是", "雖然", "很多", "還有", "有些", "繼續",
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
    # 提取核心詞
    # tokens = extract_core_words(tokens)
    # 根據需要過濾停用詞
    if not keep_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS]
    logging.debug(f"斷詞後長度 {len(tokens)}")
    return tokens


# ----------------------------------------------------------------------
# 動物詞典相關功能
# ----------------------------------------------------------------------
def load_animals(animal_file: Path = Path("animal_list.txt")) -> set:
    """從檔案載入動物詞典作為分詞的補充字典"""
    if not animal_file.exists():
        logging.error(f"找不到動物詞典檔案: {animal_file}")
        return set()
    
    with animal_file.open('r', encoding='utf-8') as f:
        animals = {line.strip() for line in f if line.strip()}
    
    # 將動物名稱添加到jieba詞典中以提高分詞準確度
    for animal in animals:
        jieba.add_word(animal)
    
    logging.debug(f"已載入 {len(animals)} 種動物詞到分詞字典")
    return animals


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
    animals = load_animals()
    
    raw_tokens = tokenize(text, keep_stopwords=True)
    filt_tokens = tokenize(text, keep_stopwords=False)

    # ---------- 停頓部份 ----------
    segs = parse_srt(srt_path)
    if not segs:
        logging.warning(f"{srt_path.name}: 找不到字幕段，停頓數將為 0")
        segs = [(0.0, 0.0)]

    actual_speaking_time = speaking_duration(segs)

    internal_gaps = [
        segs[i + 1][0] - segs[i][1]            # 段與段之間
        for i in range(len(segs) - 1)
    ]
    pauses = [g for g in internal_gaps if g > PAUSE_TH]
    total_pause_time = sum(pauses)

    # 以真實發聲時間計算 Speaking Rate
    sr = len(raw_tokens) / max(actual_speaking_time, 1.0)

    verbal_fluency = len(filt_tokens)
    lr = len(set(filt_tokens))

    # 計算加權VF
    vf_total = len(pauses)  # 原始VF（停頓總數）
    
    # 新的加權計算
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

    logging.debug(f"原始詞數 (raw_tokens): {len(raw_tokens)}")
    logging.debug(f"前 30 個原始詞: {raw_tokens[:30]}")
    logging.debug(f"去停用詞後詞數 (filtered): {verbal_fluency}")
    logging.debug(f"前 30 個過濾後詞: {filt_tokens[:30]}")
    logging.debug(f"不同詞數 (LR): {lr}")
    logging.debug(f"前 30 個不同詞: {list(set(filt_tokens))[:30]}")
    logging.debug(f"所有 gap: {pauses}")
    logging.debug(f"停頓 (> {PAUSE_TH}s) 個數: {len(pauses)}")
    logging.debug(f"各停頓長度: {pauses}")
    logging.debug(f"總停頓時間: {total_pause_time} 秒")
    logging.debug(f"實際發聲時間: {actual_speaking_time} 秒")
    logging.debug(f"停頓分布 - 輕度(2-5s): {vf_counts['level1']}, 中度(5-10s): {vf_counts['level2']}, 嚴重(>10s): {vf_counts['level3']}")
    logging.debug(f"加權後VF: {vf_weighted}")

    # 只在CSV中輸出我們需要的指標
    return {
        "speaking_rate": round(sr, 4),
        "verbal_fluency": verbal_fluency,
        "lexical_richness": lr,
        "voice_fluency_pauses": vf_total,
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
# 批次處理所有受試者
# ----------------------------------------------------------------------
import pandas as pd

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
        logging.info(f"{sid}: 完成計算 → {scores}")

    if not records:
        logging.error("整個資料夾沒有任何有效檔案！")
        return

    df = pd.DataFrame(records).set_index("subject_id")
    df.to_csv(out_csv, encoding="utf-8-sig")
    logging.info(f"已輸出彙整結果 → {out_csv}")
    print("\n============== 批次結果 ==============")
    print(df)

# ----------------------------------------------------------------------
# 直接執行批次（可視需求關掉單人測試）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TXT_DIR = Path("data/txt")
    SRT_DIR = Path("data/srt")
    OUT_CSV = Path("batch_metrics.csv")

    batch_process(TXT_DIR, SRT_DIR, OUT_CSV)
