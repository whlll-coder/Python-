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
    low_memory=False,
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
takamatsu = elec_data.join(tmp.loc[:, "temperature"]).dropna().values

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

import sklearn.model_selection
import sklearn.svm

data_count = len(takamatsu_elec)

# 交叉验证的准备(数据生成)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    takamatsu_wthr, takamatsu_elec, test_size=0.2)

# -- SVR --
model = sklearn.svm.SVR()
y_train = y_train.flatten()
y_test = y_test.flatten()

model.fit(x_train, y_train)

# -- 可视化 --
import matplotlib.pyplot as plt
import numpy as np

px = np.arange(takamatsu_wthr.min(), takamatsu_wthr.max(), 0.01)[:, np.newaxis]
py = model.predict(px)

# 生成图表
plt.xlabel('Temperature(C degree)')
plt.ylabel('electricity consumption(*10000 kW)')

plt.plot(px, py, color="black", linewidth=1,
         label='electricity consumption(predicted)')

plt.scatter(takamatsu_wthr, takamatsu_elec, s=0.5,
            color="gray", label='electricity consumption(measured)')

plt.legend(loc='upper left')
plt.savefig('7-5-2-2-graph.png')
