# coding: utf-8
import pandas as pd

# 读取四国用电的电力消费量数据
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col = "date_hour")
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

# -- 可视化 --
import matplotlib.pyplot as plt

# 设定图像的尺寸
plt.figure(figsize=(10, 6))

# 生成数据数列图表
delta = elec_data.index - pd.to_datetime('2012/07/01 00:00:00')
elec_data['time'] = delta.days + delta.seconds / 3600.0 / 24.0

plt.scatter(elec_data['time'], elec_data['consumption'], s=0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('electricity consumption(*10000 kWh)')

# 保存图表
plt.savefig('7-4-1-1-graph.png')
