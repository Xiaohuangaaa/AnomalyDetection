import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.covariance import EllipticEnvelope

def norm(data):
    # 实例化MinMaxScaler，并进行数值归一化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def downsample(data, timestamp_col, down_len):
    # 将数据按照每个时间窗口的起始时间戳分组
    grouped = data.groupby(data.index // down_len)

    downsampled_data = []
    timestamps = []

    # 遍历每个时间窗口的数据
    for _, group in grouped:
        # 提取每个时间窗口的第一个时间戳
        timestamp = group.iloc[0, timestamp_col]
        timestamps.append(timestamp)

        # 计算每个时间窗口的十分钟平均值
        sum_value = group.iloc[:, 1].sum()  # 假设数据列从第二列开始
        downsampled_data.append(sum_value)

    return timestamps, downsampled_data

def detect_outliers(data, window_size):
    # 计算移动窗口下的移动平均值
    moving_avg = data.rolling(window=window_size).mean()

    # 计算每个数据点与移动平均值之间的差异
    diff = np.abs(data - moving_avg)

    # 标记异常值
    outliers = diff > 0.8*moving_avg

    # 将异常值替换为缺失值
    data[outliers] = None

    # 使用线性插值填充缺失值
    data = data.interpolate(method='linear')

    return data

    return data


def main():
    # 读取Excel文件
    df = pd.read_csv("../data/Apt17_2016.csv")

    # 提取电力负荷列的数据
    power_loads = df.iloc[:, 1]

    df.iloc[:,1] *= 1000

    # 提取要检测的数据列
    data_to_check = df.iloc[:,1]

    # 检测异常值（假设窗口大小为10，阈值为1）
    window_size = 10
    threshold = 1
    data_without_outliers = detect_outliers(data_to_check, window_size)

    # 将处理后的数据替换负荷列
    df.iloc[:,1] = data_without_outliers

    # 归一化
    normalized_power_loads = norm(power_loads.values.reshape(-1, 1))

    # 将归一化后的值替换原始数据中的电力负荷列
    df.iloc[:, 1] = normalized_power_loads.flatten()

    # # 降采样，每10分钟一次
    # timestamps, load_downsample = downsample(df, timestamp_col=0, down_len=10)

    # 保存处理后的数据到新的CSV文件
    df.to_csv("processed_load_data_2.csv", index=False)

if __name__ == "__main__":
    main()
