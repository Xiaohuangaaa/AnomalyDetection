import pandas as pd

# 读取处理后的负荷数据
data = pd.read_csv("anomaly_data_1.csv")
data['Timestamp'] = pd.to_datetime(data.iloc[:,0])
# data['Timestamp'] = pd.to_datetime(data['Timestamp'])  # 将时间戳转换为 datetime 对象

# 读取天气数据
weather_data = pd.read_csv("../data/apartment2016.csv")

weather_hourly = weather_data.values

# 每个小时的负荷数据都对应相同的天气数据
num_load_records_per_hour = 6  # 负荷数据每小时的记录数
num_weather_records = len(weather_hourly)  # 天气数据的总记录数

# 按小时分组，为每个小时的所有数据设置相同的天气值
for i in range(0, len(data), num_load_records_per_hour):
    hour_index = i // num_load_records_per_hour
    weather_index = min(hour_index, num_weather_records - 1)  # 确保天气索引不超过天气数据长度
    data.loc[i:i+num_load_records_per_hour-1, 'temperature'] = weather_hourly[weather_index, 0]
    data.loc[i:i+num_load_records_per_hour-1, 'apparentTemperature'] = weather_hourly[weather_index, 5]
    data.loc[i:i+num_load_records_per_hour-1, 'humidity'] = weather_hourly[weather_index, 2]
    data.loc[i:i+num_load_records_per_hour-1, 'pressure'] = weather_hourly[weather_index,6]

def getWeekday(day):
    if day.weekday() < 5:  # 星期一到星期五为1到5
        return day.weekday() + 1
    elif day.weekday() ==6:     #周日
        return 0
    else:  # 星期六为6
        return 6

# 添加日期的月份、小时以及星期几特征
data['Month'] = data['Timestamp'].dt.month
data['Hour'] = data['Timestamp'].dt.hour
data['Weekday'] = data['Timestamp'].dt.weekday

# 自定义函数判断是否为工作日
def is_workday(day):
    if day.weekday() < 5:  # 星期一到星期五为工作日，weekday()函数返回0-6，其中0代表星期一，6代表星期日
        return 1
    else:
        return 0

# 添加工作日标记列
data['Is_Workday'] = data['Timestamp'].apply(is_workday)

# 保存添加天气数据后的数据
data.to_csv("anomaly_data_with_weather_2.csv", index=False)