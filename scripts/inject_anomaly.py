import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取原始数据
data = pd.read_csv("processed_load_data_2.csv")  # 假设原始数据保存在 original_data.csv 文件中
data.columns = ['Timestamp', 'Load']
data['Timestamp'] = pd.to_datetime(data.iloc[:,0])  # 将日期类型的时间戳转换为 datetime 对象
data['label'] = 0  # 添加标签列，默认为0表示正常数据


# 定义异常注入函数
def inject_anomaly(data, anomaly_type, target_ratio):
    # anomaly_type: 异常类型，可以是 "line_disconnection", "equipment_failure", "random_noise", "outliers"
    # target_ratio: 目标异常数据占全部数据的比例

    # 计算异常数据的数量
    anomaly_count = int(len(data) * target_ratio)

    if anomaly_type == "line_disconnection":
        # 模拟输电线路断开导致负荷为0
        anomaly_indices = []
        existing_anomaly_indices = data[data['label'] == 1].index
        while len(anomaly_indices) < anomaly_count:
            start_idx = np.random.randint(len(data))
            end_idx = start_idx + np.random.randint(2,15)  # 随机选择异常持续时间
            end_idx = min(end_idx, len(data))
            # 确保新的异常数据时间段与已经产生的异常数据时间段不重叠
            if not np.any(np.logical_and(start_idx <= existing_anomaly_indices, existing_anomaly_indices <= end_idx)):
                anomaly_indices.extend(list(range(start_idx, end_idx)))
        data.loc[anomaly_indices, 'Load'] = 0  # 负荷直接变为0

    elif anomaly_type == "equipment_failure":
        # 模拟家庭设备故障导致负荷波动
        anomaly_indices = []
        existing_anomaly_indices = data[data['label'] == 1].index
        while len(anomaly_indices) < anomaly_count:
            start_idx = np.random.randint(len(data))
            end_idx = start_idx + np.random.randint(2, 15)  # 随机选择异常持续时间
            end_idx = min(end_idx, len(data))
            # 确保新的异常数据时间段与已经产生的异常数据时间段不重叠
            if not np.any(np.logical_and(start_idx <= existing_anomaly_indices, existing_anomaly_indices <= end_idx)):
                anomaly_indices.extend(list(range(start_idx, end_idx)))
        data.loc[anomaly_indices, 'Load'] *= np.random.uniform(0.8, 1.2, size=len(anomaly_indices))  # 负荷波动

    elif anomaly_type == "random_noise":
        # 添加随机扰动
        scaler = StandardScaler()
        data['Load'] = scaler.fit_transform(data['Load'].values.reshape(-1, 1))
        anomaly_indices = []
        existing_anomaly_indices = data[data['label'] == 1].index
        while len(anomaly_indices) < anomaly_count:
            start_idx = np.random.randint(len(data))
            end_idx = start_idx + np.random.randint(2, 20)  # 随机选择异常持续时间
            end_idx = min(end_idx, len(data))
            # 确保新的异常数据时间段与已经产生的异常数据时间段不重叠
            if not np.any(np.logical_and(start_idx <= existing_anomaly_indices, existing_anomaly_indices <= end_idx)):
                anomaly_indices.extend(list(range(start_idx, end_idx)))
        data.loc[anomaly_indices, 'Load'] += np.random.normal(0, 0.0001, size=len(anomaly_indices))  # 根据数据范围调整幅度
        data['Load'] = scaler.inverse_transform(data['Load'].values.reshape(-1, 1)).flatten()

    elif anomaly_type == "outliers":
        # 添加孤立点异常
        anomaly_indices = []
        existing_anomaly_indices = data[data['label'] == 1].index
        while len(anomaly_indices) < anomaly_count:
            idx = np.random.randint(len(data))
            # 确保新的异常数据时间点不与已经产生的异常数据时间点重叠
            if idx not in existing_anomaly_indices:
                anomaly_indices.append(idx)
        data.loc[anomaly_indices, 'Load'] += np.random.normal(0, 0.0002, size=len(anomaly_indices))  # 根据数据范围调整幅度

    # 将异常数据标记为1，表示异常
    data.loc[anomaly_indices, 'label'] = 1


# 模拟注入异常，并确保异常数据比例满足要求
# 输电线路断开异常，目标异常数据占全部数据的1.5%
inject_anomaly(data, "line_disconnection", 0.01)
# 家庭设备故障异常，目标异常数据占全部数据的3%
inject_anomaly(data, "equipment_failure", 0.03)
# 随机扰动异常，目标异常数据占全部数据的3%
inject_anomaly(data, "random_noise", 0.03)
# 孤立点异常，目标异常数据占全部数据的1%
inject_anomaly(data, "outliers", 0.01)

# 统计标签为1的数据比例
label_1_ratio = data['label'].mean()
print("标签为1的数据比例：", label_1_ratio)

# 将注入异常后的数据保存到 CSV 文件中
data.to_csv("anomaly_data_1.csv", index=False)
