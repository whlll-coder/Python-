# coding: utf-8
import pandas as pd

# 读取气象数据
tmp = pd.read_csv(
    u'47891_高松.csv',
    parse_dates={'date_hour': ["日期"]},
    index_col="date_hour",
    na_values="×"
)

del tmp["时间"]  # ［时间］的列不使用，所以删除

# 列的名字中不要带有日语，所以仅把接下来要使用的列的名字修改成英语
columns = {
    "降水量(mm)": "rain",
    "气温(℃)": "temperature",
    "日照时间(h)": "sunhour",
    "湿度(％)": "humid",
}
tmp.rename(columns=columns, inplace=True)

# -- 可视化 --
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# 生成图表

plt.hist(tmp['temperature'], bins=50, color="gray")
plt.xlabel('Temperature(C degree)')
plt.ylabel('count')

# 保存图表
plt.savefig('7-4-1-4-graph.png')
