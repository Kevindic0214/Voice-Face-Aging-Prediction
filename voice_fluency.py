#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算 Voice Fluency（聲音流暢度）指標：
Voice Fluency = (possible_intervals - pause_count) / possible_intervals

停頓定義採用多階層劃分：
- micro        : 0.00–0.20 秒（不計為停頓）
- hesitation   : 0.20–0.50 秒（不計為停頓）
- cognitive    : 0.50–1.00 秒（計為停頓）
- phrase       : 1.00–2.00 秒（計為停頓）
- long         : >2.00       秒（計為停頓）
"""

import logging
import re
import argparse
from pathlib import Path
from typing import List, Tuple

# 總錄音長度（秒）
WINDOW_SEC = 59.0

# SRT 時間戳正則
_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")

# 停頓分類閾值（秒）
PAUSE_LEVELS = {
    "micro":      (0.00, 0.20),
    "hesitation": (0.20, 0.50),
    "cognitive":  (0.50, 1.00),
    "phrase":     (1.00, 2.00),
    "long":       (2.00, float("inf")),
}

# 哪些級別算「實際停頓」
PAUSE_CATEGORIES_COUNTED = {"cognitive", "phrase", "long"}


def ts2sec(ts: str) -> float:
    """HH:MM:SS,mmm → 秒 (float)"""
    m = _TIME_RE.match(ts)
    if not m:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, m_, s, ms = map(int, m.groups())
    return h * 3600 + m_ * 60 + s + ms / 1000.0


def parse_srt(path: Path) -> List[Tuple[float, float]]:
    """解析 SRT，回傳所有 (start, end) 時間段"""
    segs: List[Tuple[float, float]] = []
    with path.open(encoding="utf-8") as fh:
        mode = 0
        for line in fh:
            line = line.strip()
            if mode == 0 and line.isdigit():
                mode = 1
            elif mode == 1 and "-->" in line:
                start_ts, end_ts = [p.strip() for p in line.split("-->")]
                segs.append((ts2sec(start_ts), ts2sec(end_ts)))
                mode = 2
            elif mode == 2 and line == "":
                mode = 0
    logging.debug(f"{path.name}: parsed {len(segs)} segments")
    return segs


def classify_pause(duration: float) -> str:
    """根據 PAUSE_LEVELS 返回停頓級別名稱"""
    for level, (mn, mx) in PAUSE_LEVELS.items():
        if mn <= duration < mx:
            return level
    return "unknown"


def calculate_voice_fluency(srt_path: Path) -> Tuple[float, dict]:
    """計算 Voice Fluency 及各級停頓次數"""
    segs = parse_srt(srt_path)
    # 若完全無語音段，視為流暢度 0
    if not segs:
        return 0.0, {lvl: 0 for lvl in PAUSE_LEVELS}

    # 收集所有停頓時長
    pauses: List[float] = []
    # 開頭
    first_gap = segs[0][0]
    pauses.append(first_gap)
    # 段間
    for i in range(len(segs) - 1):
        pauses.append(segs[i+1][0] - segs[i][1])
    # 結尾
    pauses.append(WINDOW_SEC - segs[-1][1])

    # 統計各級別次數
    counts = {lvl: 0 for lvl in PAUSE_LEVELS}
    for p in pauses:
        lvl = classify_pause(p)
        if lvl in counts:
            counts[lvl] += 1

    # 實際停頓次數＝計入的級別數量
    pause_count = sum(counts[lvl] for lvl in PAUSE_CATEGORIES_COUNTED)
    possible_intervals = len(segs) + 1

    vf = (possible_intervals - pause_count) / possible_intervals
    vf = max(0.0, min(1.0, vf))

    return vf, counts


def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Compute Voice Fluency from an SRT file using multi-level pause definitions"
    )
    parser.add_argument("srt", type=Path, help="Path to the .srt file")
    args = parser.parse_args()

    if not args.srt.exists():
        logging.error(f"{args.srt} does not exist.")
        return

    vf_score, pause_counts = calculate_voice_fluency(args.srt)
    print(f"Voice Fluency: {vf_score:.4f}")
    print("Pause counts by level:")
    for lvl, cnt in pause_counts.items():
        print(f"  {lvl:11s}: {cnt}")


if __name__ == "__main__":
    main()
