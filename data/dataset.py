import pandas as pd
from sklearn.model_selection import train_test_split

# 读取包含天气数据的负荷数据
data = pd.read_csv("../scripts/anomaly_data_with_weather_2.csv")  # 假设处理后的数据保存在 anomaly_data_with_weather.csv 文件中
data['Timestamp'] = pd.to_datetime(data['Timestamp'])  # 将时间戳转换为 datetime 对象

# 计算数据集的长度
total_length = len(data)

# 计算每个季度的数据量
quarter_length = total_length // 4

# 将数据按季度分组
grouped_data = data.groupby(data['Timestamp'].dt.quarter)

# 初始化训练集和测试集
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# 统计训练集和测试集中每个季度的数据量
train_quarter_counts = {}
test_quarter_counts = {}

# 对每个季度的数据进行处理
for quarter, group in grouped_data:
    # 计算分界点
    split_index = int(len(group) * 0.7)

    # 将前70%的数据放入训练集，后30%的数据放入测试集
    train_data = pd.concat([train_data, group.head(split_index)])
    test_data = pd.concat([test_data, group.tail(len(group) - split_index)])

    # 统计每个季度的数据量
    train_quarter_counts[quarter] = len(train_data[train_data['Timestamp'].dt.quarter == quarter])
    test_quarter_counts[quarter] = len(test_data[test_data['Timestamp'].dt.quarter == quarter])

# 输出训练集和测试集中每个季度的数据量
for quarter in range(1, 5):
    print(f"训练集季度 {quarter} 的数据条数为: {train_quarter_counts[quarter]}")
    print(f"测试集季度 {quarter} 的数据条数为: {test_quarter_counts[quarter]}")

# train_size = quarter_length * 3  # 使用前三个季度的数据作为训练集
# test_size = total_length - train_size  # 使用最后一个季度的数据作为测试集


# # 删除 train_data 中的 Timestamp 列
# train_data.drop(columns=['Timestamp'], inplace=True)
#
# # 删除 test_data 中的 Timestamp 列
# test_data.drop(columns=['Timestamp'], inplace=True)


# # 统计标签为1的数据比例
# label_1_ratio = data1['label'].mean()
# print("trainset标签为1的数据比例：", label_1_ratio)
#
# data2 = pd.read_csv('test_data.csv')
# data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
# label_2_ratio = data2['label'].mean()
# print("标签为1的数据比例：", label_2_ratio)
# # 划分训练集和测试集
# # 按时间顺序划分，保证测试集中的时间不早于训练集中的时间
# train_size = 0.7  # 训练集占比
# split_index = int(len(data) * train_size)
# train_data = data.iloc[:split_index]
# test_data = data.iloc[split_index:]

train_data.drop(columns=['Timestamp'],inplace= True)
test_data.drop(columns=['Timestamp'],inplace=True)

# 保存划分好的训练集和测试集
train_data.to_csv("train_data.csv", index=True, index_label="index")
test_data.to_csv("test_data.csv", index=True, index_label="index")
