import os

# 設定你的資料夾路徑
folder_path = r'C:\kevin\Voice-Face-Aging-Prediction\逐字稿'

# 走訪資料夾內所有 .txt 檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        if first_line:  # 確保第一行不是空的
            # 組合新檔名
            new_filename = first_line + '.txt'
            new_filepath = os.path.join(folder_path, new_filename)

            # 如果檔名已存在，加上編號避免覆蓋
            counter = 1
            base_name = first_line
            while os.path.exists(new_filepath):
                new_filename = f"{base_name}_{counter}.txt"
                new_filepath = os.path.join(folder_path, new_filename)
                counter += 1

            os.rename(file_path, new_filepath)
            print(f"✅ {filename} → {new_filename}")
        else:
            print(f"⚠️ 檔案 {filename} 的第一行是空白，略過。")
