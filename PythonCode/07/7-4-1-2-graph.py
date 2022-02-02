# coding: utf-8
import pandas as pd

# 读取电力消费量数据
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col='date_hour')
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

# -- 可视化 --
import matplotlib.pyplot as plt

# 设定图像的尺寸
plt.figure(figsize=(10, 6))

# 生成柱状图
plt.hist(elec_data['consumption'], bins=50, color="gray")
plt.xlabel('electricity consumption(*10000 kW)')
plt.ylabel(u'count')

# 保存图表
plt.savefig('7-4-1-2-graph.png')
