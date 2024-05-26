import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/test_data.csv')

# 标准化处理
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['Load', 'temperature', 'apparentTemperature', 'humidity', 'pressure', 'Month', 'Hour', 'Weekday', 'Is_Workday', 'label']])
test_scaled = scaler.transform(test_data[['Load', 'temperature', 'apparentTemperature', 'humidity', 'pressure', 'Month', 'Hour', 'Weekday', 'Is_Workday','label']])

# 合并标签列到数据集中
train_scaled = np.hstack((train_scaled, train_data['label'].values.reshape(-1, 1)))
test_scaled = np.hstack((test_scaled, test_data['label'].values.reshape(-1, 1)))

# 准备训练和测试数据
def create_sequences(data, time_steps=1):
    xs, ys = [], []
    for i in range(len(data) - time_steps):
        xs.append(data[i:i + time_steps, :-1])  # 所有特征
        ys.append(data[i + time_steps, -1])     # 标签
    return np.array(xs), np.array(ys)

time_steps = 60

X_train, y_train = create_sequences(train_scaled, time_steps)
X_test, y_test = create_sequences(test_scaled, time_steps)


# 构建LSTM模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, shuffle=False)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 转换为二进制类别
y_pred_train = (y_pred_train > 0.5).astype(int)
y_pred_test = (y_pred_test > 0.5).astype(int)

# 评估性能
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 保存实际标签和预测结果为 .npy 文件
np.save('LSTM_y_test.npy', y_test)
np.save('LSTM_y_pred_test.npy', y_pred_test)


# 绘制结果
# plt.figure(figsize=(15, 5))
#
# plt.plot(test_data['label'][time_steps:].values, color='blue', label='Actual Label')
# plt.plot(np.arange(time_steps, len(test_data)), y_pred_test, color='orange', label='Predicted Label')
#
# plt.legend()
# plt.show()

# 检查实际标签和预测标签的分布
unique, counts = np.unique(y_test, return_counts=True)
print(f'Actual Label Distribution: {dict(zip(unique, counts))}')

unique, counts = np.unique(y_pred_test, return_counts=True)
print(f'Predicted Label Distribution: {dict(zip(unique, counts))}')

# 绘制结果
plt.figure(figsize=(15, 5))

# 只绘制前1000个数据点
plot_range = 10000

plt.plot(y_test[:plot_range], color='blue', label='Actual Label')
plt.plot(y_pred_test[:plot_range], color='orange', linestyle='dashed', label='Predicted Label')

plt.legend()
plt.show()
