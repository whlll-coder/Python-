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
    na_values=["×", "--"]
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
tmp.fillna(-1,inplace=True)

# 月, 日, 时间的取得
tmp["month"] = tmp.index.month
tmp['day'] = tmp.index.day
tmp['dayofyear'] = tmp.index.dayofyear
tmp['hour'] = tmp.index.hour

# 将气象数据和电力消费量数据暂且统和并且和时间轴整合，之后再重新划分
takamatsu = elec_data.join(tmp[["temperature","sunhour","month","hour"]]).dropna().values

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

# 学习和性能的评价
import sklearn.model_selection
import sklearn.svm
model = sklearn.svm.SVR()



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    takamatsu_wthr, takamatsu_elec, test_size=0.2)

y_train = y_train.flatten()
y_test = y_test.flatten()

model.fit(x_train, y_train)
date_name = ["气温", "日照时间","月","时间"]

output = "使用项目 = %s, 训练分数= %f, 校验评分= %f" % \
         (", ".join(date_name),
          model.score(x_train, y_train),
          model.score(x_test, y_test)
          )
#    print (output.decode('utf-8')) # Python2的话请使用这一行
print (output)  # 面向Python 3



# -- 可视化 --
import matplotlib.pyplot as plt

# 设定图像大小
plt.figure(figsize=(10, 6))

predicted = model.predict(x_test)

plt.xlabel('electricity consumption(measured *10000 kW)')
plt.ylabel('electricity consumption(predicted *10000 kW)')
plt.scatter(y_test, predicted, s=0.5, color="black")

plt.savefig("7-5-5-4-graph.png")


