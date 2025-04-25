import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------- 檔案路徑 ----------
BATCH_CSV = Path("batch_metrics.csv")
LABEL_CSV = Path("labels.csv")
PLOT_DIR  = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# ---------- 讀檔與合併 ----------
batch_df  = pd.read_csv(BATCH_CSV, dtype={"subject_id": str})
labels_df = pd.read_csv(LABEL_CSV, dtype={"NUMBER": str})
labels_df["subject_id"] = labels_df["NUMBER"].astype(int).apply(lambda x: f"{x:04d}")
labels_df.dropna(subset=["Analysis Result"], inplace=True)
labels_df["Analysis Result"] = labels_df["Analysis Result"].astype(int)

df = labels_df.merge(batch_df, on="subject_id", how="inner")
if df.empty:
    raise SystemExit("找不到任何可對應的受試者，請檢查 subject_id / NUMBER 是否一致")
logging.info(f"合併後筆數：{len(df)}")

# ---------- 指標對應 ----------
score_cols = {
    # 原始指標
    "speaking_rate":        "Speaking Rate (words/sec)",
    "verbal_fluency":       "Verbal Fluency (word count)",
    "lexical_richness":     "Lexical Richness (unique words)",
    
    "voice_fluency_weighted": "Voice Fluency (weighted)",
    "pause_level1_count": "Pause count (2-5s, Level 1)",
    "pause_level2_count": "Pause count (5-10s, Level 2)",
    "pause_level3_count": "Pause count (>10s, Level 3)",
    "total_pause_time": "Total Pause Time (sec)",
}
label_map = {0: "Normal", 1: "Dementia"}

# ---------- 畫平滑曲線圖 ----------
for col, title in score_cols.items():
    # 檢查欄位是否存在
    if col not in df.columns:
        logging.warning(f"欄位 {col} 不存在於數據中，跳過繪圖")
        continue
        
    plt.figure(figsize=(5, 4))
    
    # 設置更好的顏色
    colors = ['#3498db', '#e74c3c']  # 藍色和紅色，更美觀
    
    for i, (label, grp) in enumerate(df.groupby("Analysis Result")):
        data = grp[col].dropna()
        if len(data) < 2:  # 至少需要2個數據點
            continue
        try:
            # 使用核密度估計繪製平滑曲線
            import scipy.stats as stats
            density = stats.gaussian_kde(data)
            
            # 設定適當的x軸範圍
            if col == "animal_accuracy":
                # 動物準確率範圍是0-1
                x_min, x_max = 0, 1
            else:
                # 其他指標使用數據範圍，稍微擴展以美觀
                x_min = max(0, data.min() - data.std() * 0.5)  # 避免負值
                x_max = data.max() + data.std() * 0.5
            
            x = np.linspace(x_min, x_max, 200)  # 增加點數使曲線更平滑
            
            plt.plot(x, density(x), 
                    linewidth=2.5,  # 稍微增加線寬
                    color=colors[i],
                    label=f"{label_map.get(label, str(label))} (n={len(data)})")
            # 填充曲線下方區域
            plt.fill_between(x, density(x), alpha=0.2, color=colors[i])
            
            # 標示均值
            mean_val = data.mean()
            plt.axvline(x=mean_val, color=colors[i], linestyle='--', alpha=0.7)
            
            # 避免標註重疊
            y_pos = density(mean_val).max() * (0.9 if i == 0 else 0.7)
            
            plt.text(mean_val, y_pos, 
                    f"Mean: {mean_val:.2f}", 
                    color=colors[i], 
                    ha='center', 
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
        except Exception as e:
            logging.warning(f"繪製 {label_map.get(label, str(label))} {col} 曲線時出錯: {e}")
    
    # 設置更好看的坐標軸
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"{title} Distribution", fontsize=14, fontweight='bold')
    plt.xlabel(title, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Analysis Result")
    plt.tight_layout()
    
    # 使用曲線圖後綴名
    out_path = PLOT_DIR / f"{col}_kde_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"已輸出 {out_path}")

# ---------- 繪製停頓分佈長條圖 ----------
# 準備停頓分佈數據
pause_levels_df = pd.DataFrame()
for label, grp in df.groupby("Analysis Result"):
    if all(col in df.columns for col in ["pause_level1_count", "pause_level2_count", "pause_level3_count"]):
        group_name = label_map.get(label, str(label))
        # 計算平均值
        pause_level_means = {
            "Level 1 (2-5s)": grp["pause_level1_count"].mean(),
            "Level 2 (5-10s)": grp["pause_level2_count"].mean(),
            "Level 3 (>10s)": grp["pause_level3_count"].mean()
        }
        pause_levels_df[group_name] = pd.Series(pause_level_means)

if not pause_levels_df.empty:
    plt.figure(figsize=(15, 8))
    pause_levels_df.plot(kind="bar", rot=0)
    plt.title("Average Pause Count by Level")
    plt.xlabel("Pause Level")
    plt.ylabel("Average Count")
    plt.legend(title="Analysis Result")
    plt.tight_layout()
    out_path = PLOT_DIR / f"pause_levels_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"已輸出 {out_path}")

logging.info("所有分布圖繪製完成")
