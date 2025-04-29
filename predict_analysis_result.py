import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 讀取資料
print("讀取資料...")
features_df = pd.read_csv('batch_metrics.csv')
labels_df = pd.read_csv('labels.csv')

# 處理資料
print("處理資料...")
# 檢查資料格式問題
print("features_df['subject_id'] dtype:", features_df['subject_id'].dtype)
print("labels_df['NUMBER'] dtype:", labels_df['NUMBER'].dtype)
print("features_df['subject_id'] 前10個值:", features_df['subject_id'].head(10).values)
print("labels_df['NUMBER'] 前10個值:", labels_df['NUMBER'].head(10).values)

# 在 features_df 中，subject_id 格式為數字，需要添加前導零
features_df['subject_id_str'] = features_df['subject_id'].astype(str).str.zfill(5)
labels_df['NUMBER_str'] = labels_df['NUMBER'].astype(str).str.zfill(5)

print("修改後 features_df['subject_id_str'] 前10個值:", features_df['subject_id_str'].head(10).values)
print("修改後 labels_df['NUMBER_str'] 前10個值:", labels_df['NUMBER_str'].head(10).values)

# 移除 Analysis Result 中的空值
labels_df = labels_df.dropna(subset=['Analysis Result'])

# 檢查兩個數據集是否有匹配的ID
print(f"features_df 中的行數: {features_df.shape[0]}")
print(f"labels_df 中的行數: {labels_df.shape[0]}")

# 看看有多少 ID 是共同的
common_ids = set(features_df['subject_id_str']).intersection(set(labels_df['NUMBER_str']))
print(f"共同的 ID 數量: {len(common_ids)}")
if len(common_ids) > 0:
    print(f"共同 ID 的前10個範例: {list(common_ids)[:10]}")

# 將整數ID添加到兩個數據框中以便合併
features_df['subject_id_int'] = features_df['subject_id'].astype(int)
labels_df['NUMBER_int'] = labels_df['NUMBER'].astype(int)

# 嘗試使用整數值合併
merged_df = pd.merge(features_df, labels_df, left_on='subject_id_int', right_on='NUMBER_int', how='inner')
print(f"合併後的資料: {merged_df.shape[0]} 行")

# 如果合併後還是0行，深入檢查資料格式問題
if merged_df.shape[0] == 0:
    # 檢查並印出更多資訊
    print("\n深入檢查資料格式問題...")
    print("features_df['subject_id'] 的唯一值數量:", features_df['subject_id_int'].nunique())
    print("labels_df['NUMBER'] 的唯一值數量:", labels_df['NUMBER_int'].nunique())
    
    # 嘗試更直接的方法來找出匹配問題
    for i in range(min(10, len(features_df))):
        subject_id = features_df.iloc[i]['subject_id_int']
        found = subject_id in labels_df['NUMBER_int'].values
        print(f"subject_id {subject_id} 在 labels_df 中存在: {found}")
    
    # 看看 labels_df 中的 NUMBER 是什麼格式
    for i in range(min(10, len(labels_df))):
        number = labels_df.iloc[i]['NUMBER_int']
        found = number in features_df['subject_id_int'].values
        print(f"NUMBER {number} 在 features_df 中存在: {found}")

    # 印出兩個資料集中的所有唯一ID的範圍
    print(f"\nfeatures_df subject_id 範圍: {features_df['subject_id_int'].min()} 到 {features_df['subject_id_int'].max()}")
    print(f"labels_df NUMBER 範圍: {labels_df['NUMBER_int'].min()} 到 {labels_df['NUMBER_int'].max()}")
    
    # 終止程式
    import sys
    sys.exit("無法匹配資料，請檢查資料集中的ID!")

# 如果合併成功，則繼續
# 特徵選擇 (使用 batch_metrics.csv 中的所有特徵)
feature_columns = [
    'speech_rate', 'speech_rate_ratio', 'verbal_fluency', 'lexical_richness',
    'word_repetition_score', 'voice_fluency_pauses', 'voice_fluency_weighted',
    'pause_level1_count', 'pause_level2_count', 'pause_level3_count',
    'total_pause_time', 'total_speaking_time'
]

X = merged_df[feature_columns]
y = merged_df['Analysis Result']  # 從 labels.csv 中獲取目標變數

# 檢查類別分布
class_counts = y.value_counts()
print("類別分布:")
print(class_counts)

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 使用StratifiedKFold進行交叉驗證，確保每個折中類別比例一致
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 建立模型比較字典
models = {}

# 1. 隨機森林模型 (Random Forest)
print("\n訓練隨機森林模型...")
rf_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [100],  # 固定樹的數量
    'classifier__max_depth': [10, 20],  # 減少深度選項
    'classifier__min_samples_split': [5],  # 固定分割參數
    'classifier__min_samples_leaf': [2],  # 固定葉節點參數
    'classifier__max_features': ['sqrt'],  # 使用默認值
    'classifier__bootstrap': [True],  # 保持默認
    'classifier__oob_score': [True],  # 保持默認
    'classifier__ccp_alpha': [0.01],  # 固定正則化參數
    'smote__k_neighbors': [5]  # 固定SMOTE參數
}

# 使用RandomizedSearchCV來代替GridSearchCV，大幅減少搜索時間
n_iter = 5  # 隨機搜索的迭代次數
rf_search = RandomizedSearchCV(
    estimator=rf_pipeline, 
    param_distributions=rf_param_grid, 
    n_iter=n_iter,  # 隨機嘗試的參數組合數
    cv=skf,  # 使用StratifiedKFold進行交叉驗證
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1,
    random_state=42
)
rf_search.fit(X_train, y_train)

# 獲取最佳RF模型
best_rf_model = rf_search.best_estimator_
print(f"RF最佳參數: {rf_search.best_params_}")
models['Random Forest'] = best_rf_model

# 2. XGBoost模型
print("\n訓練XGBoost模型...")
xgb_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42))
])

xgb_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [3, 6],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8],
    'classifier__reg_alpha': [0.01],  # L1正則化
    'classifier__reg_lambda': [1.0],  # L2正則化
    'smote__k_neighbors': [5]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline, 
    param_distributions=xgb_param_grid, 
    n_iter=n_iter,
    cv=skf,
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1,
    random_state=42
)
xgb_search.fit(X_train, y_train)

# 獲取最佳XGB模型
best_xgb_model = xgb_search.best_estimator_
print(f"XGB最佳參數: {xgb_search.best_params_}")
models['XGBoost'] = best_xgb_model

# 評估所有模型
print("\n所有模型性能比較:")
results = {}
colors = {'Random Forest': 'blue', 'XGBoost': 'green'}

# 創建一個子圖，包含ROC曲線和PR曲線
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 用於存儲每個模型的評估指標
model_metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': [], 'AP': []}

for name, model in models.items():
    # 預測和概率
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 第二列是正類的概率
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    
    # 獲取分類報告並解析
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Dementia'], output_dict=True)
    precision = cr['Dementia']['precision']
    recall = cr['Dementia']['recall']
    f1 = cr['Dementia']['f1-score']
    
    # 繪製ROC曲線
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})', color=colors[name])
    
    # 繪製PR曲線
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    ax2.plot(recall_curve, precision_curve, lw=2, label=f'{name} (AP = {average_precision:.3f})', color=colors[name])
    
    # 打印評估指標
    print(f"\n{name} 模型評估:")
    print(f"準確率: {accuracy:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"平均精確率: {average_precision:.4f}")
    print("分類報告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Dementia']))
    
    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Dementia'], yticklabels=['Normal', 'Dementia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()
    
    # 保存模型指標
    model_metrics['Model'].append(name)
    model_metrics['Accuracy'].append(accuracy)
    model_metrics['Precision'].append(precision)
    model_metrics['Recall'].append(recall)
    model_metrics['F1'].append(f1)
    model_metrics['AUC'].append(roc_auc)
    model_metrics['AP'].append(average_precision)
    
    # 如果是RF，獲取特徵重要性
    if name == 'Random Forest':
        final_rf_model = model.named_steps['classifier']
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': final_rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance of Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_rf.png')
        plt.close()
    
    # 如果是XGBoost，獲取特徵重要性
    if name == 'XGBoost':
        final_xgb_model = model.named_steps['classifier']
        # 獲取特徵重要性
        xgb_importance = final_xgb_model.feature_importances_
        xgb_feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': xgb_importance
        })
        xgb_feature_importance = xgb_feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=xgb_feature_importance)
        plt.title('Feature Importance of XGBoost Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_xgb.png')
        plt.close()

# 設置ROC曲線圖
ax1.plot([0, 1], [0, 1], 'k--', lw=2)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
ax1.legend(loc="lower right")
ax1.grid(True, linestyle='--', alpha=0.7)

# 設置PR曲線圖
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_curves.png')
plt.close()

# 創建模型比較條形圖
metrics_df = pd.DataFrame(model_metrics)
plt.figure(figsize=(12, 8))
metrics_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Value')
sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_melted)
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

# 選擇最佳模型（基於AUC）
best_model_idx = metrics_df['AUC'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'Model']
best_model = models[best_model_name]
print(f"\n基於AUC選擇的最佳模型: {best_model_name}")

# 對全部資料進行預測（使用最佳模型）
print("\n對全部資料進行預測...")
# 讀取所有labels.csv中的資料（包括未參與訓練的資料）
all_labels_data = pd.read_csv('labels.csv')

# 保留原始的Analysis Result作為參考
all_labels_data['Original_Analysis_Result'] = all_labels_data['Analysis Result']

# 將NUMBER轉換為整數類型以便合併
all_labels_data['NUMBER_int'] = all_labels_data['NUMBER'].astype(int)

# 找出在batch_metrics.csv中有對應特徵的subject_id
all_merged_data = pd.merge(features_df, all_labels_data, left_on='subject_id_int', right_on='NUMBER_int', how='inner')

# 對合併後的資料進行預測
all_X = all_merged_data[feature_columns]
# 使用pipeline進行預測（它會自動處理標準化和其他步驟）
all_predictions = best_model.predict(all_X)
all_proba = best_model.predict_proba(all_X)

# 將預測結果添加到資料框
all_merged_data['Predicted_Result'] = all_predictions
all_merged_data['Predicted_Result_Label'] = all_merged_data['Predicted_Result'].apply(lambda x: 'Dementia' if x == 1.0 else 'Normal')
all_merged_data['Prediction_Confidence'] = np.max(all_proba, axis=1)

# 計算預測結果與原始結果的一致性（排除原始資料中的空值）
mask = ~all_merged_data['Original_Analysis_Result'].isna()
all_merged_data.loc[mask, 'Is_Correct'] = (all_merged_data.loc[mask, 'Predicted_Result'] == all_merged_data.loc[mask, 'Original_Analysis_Result'])
correct_count = all_merged_data.loc[mask, 'Is_Correct'].sum()
total_count = mask.sum()
print(f"預測結果與原始結果一致率: {correct_count/total_count:.4f} ({correct_count}/{total_count})")

# 保存預測結果
results_df = all_merged_data[['NUMBER', 'Original_Analysis_Result', 'Predicted_Result_Label', 'Prediction_Confidence', 'Is_Correct']]
results_df.to_csv('prediction_results.csv', index=False)
print("預測結果已保存至 prediction_results.csv") 