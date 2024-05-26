# import SMOTE as SMOTE
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. 加载数据
train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/test_data.csv')

# 2. 选择特征
features = ['Load', 'temperature', 'apparentTemperature', 'humidity', 'pressure', 'Month', 'Hour', 'Weekday', 'Is_Workday']
X_train = train_data[features].values
y_train = train_data['label'].values  # 使用 label 列作为真实标签
X_test = test_data[features].values
y_test = test_data['label'].values

# # 3. 数据平衡处理
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# 4. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 训练KNN模型
knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_model.fit(X_train_scaled)

# 6. 在测试集上进行异常检测
distances, indices = knn_model.kneighbors(X_test_scaled)
distances_mean = distances.mean(axis=1)

# 定义异常阈值，例如基于平均距离的一个倍数
threshold = 2.0 * distances_mean.std()

# 标记异常点
test_outliers = distances_mean > threshold

# 使用阈值结果作为预测标签
y_pred = test_outliers.astype(int)

# 7. 评估模型性能
# 打印分类报告和混淆矩阵
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 计算 Precision, Recall 和 F1 分数
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# # 8. 可视化测试集上的结果
# plt.figure(figsize=(12, 6))
# plt.plot(test_data.index[-len(y_test):], test_data['Load'][-len(y_test):], label='Load')
#
# # 真实的异常点
# true_anomalies = (y_test == 1)
# plt.scatter(test_data.index[-len(y_test):][true_anomalies], test_data['Load'][-len(y_test):][true_anomalies], color='g', label='True Anomalies')
#
# # 预测的异常点
# predicted_anomalies = test_outliers
# plt.scatter(test_data.index[-len(y_test):][predicted_anomalies], test_data['Load'][-len(y_test):][predicted_anomalies], color='r', marker='x', label='Predicted Anomalies')
#
# # 同时是实际异常点又是预测异常点
# correct_anomalies = true_anomalies & predicted_anomalies
# plt.scatter(test_data.index[-len(y_test):][correct_anomalies], test_data['Load'][-len(y_test):][correct_anomalies], color='b', marker='o', label='Correctly Predicted Anomalies')
#
# plt.xlabel('Time')
# plt.ylabel('Load')
# plt.title('Electricity Load Anomaly Detection on Test Set')
# plt.legend()
# plt.show()

subset_size = 10000  # 选择一个适当的子集大小
subset_indices = slice(0, subset_size)

plt.figure(figsize=(12, 6))
plt.plot(test_data.index[-len(y_test):][subset_indices], test_data['Load'][-len(y_test):][subset_indices], label='Load')

# 真实的异常点
true_anomalies = (y_test == 1)
plt.scatter(test_data.index[-len(y_test):][subset_indices][true_anomalies[subset_indices]], test_data['Load'][-len(y_test):][subset_indices][true_anomalies[subset_indices]], color='g', label='True Anomalies')

# 预测的异常点
predicted_anomalies = test_outliers
plt.scatter(test_data.index[-len(y_test):][subset_indices][predicted_anomalies[subset_indices]], test_data['Load'][-len(y_test):][subset_indices][predicted_anomalies[subset_indices]], color='r', marker='x', label='Predicted Anomalies')

# 同时是实际异常点又是预测异常点
correct_anomalies = true_anomalies & predicted_anomalies
plt.scatter(test_data.index[-len(y_test):][subset_indices][correct_anomalies[subset_indices]], test_data['Load'][-len(y_test):][subset_indices][correct_anomalies[subset_indices]], color='b', marker='o', label='Correctly Predicted Anomalies')

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Electricity Load Anomaly Detection on Test Set (Subset)')
plt.legend()
plt.show()