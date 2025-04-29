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
    "speech_rate":        "Speech Rate (words/sec)",
    "speech_rate_ratio":   "Speech Rate Ratio (first_half/second_half)",
    "verbal_fluency":       "Verbal Fluency (word count)",
    "lexical_richness":     "Lexical Richness (unique words)",
    "word_repetition_score": "Word Repetition Score",
    
    "voice_fluency_weighted": "Voice Fluency (weighted)",
    "pause_level1_count": "Pause count (2-5s, Level 1)",
    "pause_level2_count": "Pause count (5-10s, Level 2)",
    "pause_level3_count": "Pause count (>10s, Level 3)",
    "total_pause_time": "Total Pause Time (sec)",
}
label_map = {0: "Normal", 1: "Dementia"}

# ---------- 畫平滑曲線圖 ----------
for col, title in score_cols.items():
    if col not in df.columns:
        continue
        
    plt.figure(figsize=(6, 4))
    
    # 動態語速比專屬設定
    if col == "speech_rate_ratio":
        # 坐標軸範圍設定
        plt.xlim(0.3, 2.5)
        
        # 參考線與解讀標註
        plt.axvline(x=1.0, color='gray', linestyle=':', 
                   label='Baseline (no change)')
        plt.axvspan(1.3, 2.5, alpha=0.1, color='red', 
                   label='Possible impairment')
        
        # 顯著性標記（需預先統計檢定）
        for i, (label, grp) in enumerate(df.groupby("Analysis Result")):
            if grp[col].mean() > 1.3:  # 假設1.3為閾值
                y_pos = 0.9 - i*0.1
                plt.text(1.8, y_pos, '*p<0.05', 
                        fontsize=12, ha='center')

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

# ---------- 繪製各分數指標箱型圖 ----------
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']  # 支援中文顯示
plt.rcParams['axes.unicode_minus'] = False  # 讓負號正確顯示

for col, title in score_cols.items():
    # 檢查欄位是否存在
    if col not in df.columns:
        logging.warning(f"欄位 {col} 不存在於數據中，跳過繪製箱型圖")
        continue
        
    plt.figure(figsize=(8, 6))
    
    # 設置更好的顏色
    colors = ['#3498db', '#e74c3c']  # 藍色和紅色
    
    boxplot = df.boxplot(column=col, by="Analysis Result", patch_artist=True, 
                        return_type='dict', figsize=(8, 6))
    
    # 美化箱型圖
    for i, box in enumerate(boxplot[col]['boxes']):
        box.set(facecolor=colors[i], alpha=0.7)
    
    # 添加數據點
    for i, (label, grp) in enumerate(df.groupby("Analysis Result")):
        # 計算抖動量
        jitter = 0.05
        x = np.random.normal(i+1, jitter, size=len(grp))
        plt.scatter(x, grp[col], alpha=0.6, s=30, color=colors[i], edgecolor='k')
    
    # 透過設置title和suptitle移除自動生成的標題
    plt.suptitle('')
    plt.title(f"{title} Boxplot by Analysis Result", fontsize=14, fontweight='bold')
    plt.xlabel("Analysis Result", fontsize=12)
    plt.ylabel(title, fontsize=12)
    
    # 修改x軸標籤
    ax = plt.gca()
    # 直接使用 Analysis Result 的值作為標籤
    labels = [label_map.get(label, 'Unknown') for label in df['Analysis Result'].unique()]
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    
    # 保存箱型圖
    out_path = PLOT_DIR / f"{col}_boxplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"已輸出 {out_path}")

logging.info("所有箱型圖繪製完成")
