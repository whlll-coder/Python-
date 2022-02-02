# coding: utf-8
import pandas as pd

# 读取四国电力的用电量
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col='date_hour')
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

# 读取气象数据
tmp = pd.read_csv(
    u'47891_高松.csv',
    parse_dates={'date_hour': ["日期"]},
    index_col="date_hour",
    na_values="×"
)

del tmp["时间"]  # 由于不使用［时间］的列，所以删除

# 列的名字中含有日语不好，所以仅把接下来要使用的列的名字修改成英语
columns = {
    "降水量(mm)": "rain",
    "气温(℃)": "temperature",
    "日照时间(h)": "sunhour",
    "湿度(％)": "humid",
}
tmp.rename(columns=columns, inplace=True)

# 将气象数据和电力消费量数据暂且统和并且和时间轴整合，之后再重新划分
takamatsu = elec_data.join(tmp["temperature"]).dropna().values

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_whhr = takamatsu[:, 1:]

# -- 可视化 --
import matplotlib.pyplot as plt

# 生成图表
plt.xlabel('Temperature(C degree)')
plt.ylabel('electricity consumption(*10000 kW)')

# 环境设定
plt.scatter(takamatsu_whhr, takamatsu_elec, s=0.5,
            color="gray", label='electricity consumption(measured)')

plt.savefig('7-5-1-1-graph.png')
