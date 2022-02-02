# coding: utf-8
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

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

tmp_org = tmp

for h_count in range(1,6):
    print("\n h_count:"+str(h_count))
    
    tmp = tmp_org[["temperature","sunhour"]]
    ld = tmp

    for i in range(1,h_count):
        ld = ld.join(tmp.shift(i),rsuffix="_"+str(i)).dropna()

    tmp = ld
    ## 合并数据 
    takamatsu = elec_data.join(tmp).dropna().values
    
    takamatsu_elec = takamatsu[:, 0:1]
    takamatsu_wthr = takamatsu[:, 1:]
    
    # 学习和性能的评价
    import sklearn.model_selection
    import sklearn.svm
    
    data_count = takamatsu_elec
    
    # 交叉验证的准备
    kf = sklearn.model_selection.KFold(n_splits=5)
    
    # 实施交叉验证 (实施所有模式)
    for train, test in kf.split(data_count):
        x_train = takamatsu_wthr[train]
        x_test = takamatsu_wthr[test]
        y_train = takamatsu_elec[train]
        y_test = takamatsu_elec[test]
    
        # -- SVR --
        model = sklearn.svm.SVR()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    
        model.fit(x_train, y_train)
        print ("SVR: Training Score = %f, Testing(Validate) Score = %f" %
               (model.score(x_train, y_train), model.score(x_test, y_test)))
