关于《Python机器学习入门 第7章 传感器数据的回归问题》代码说明
2016-11-28 ISP

[使用方法]
・请将数据文件（CSV文件）放在与脚本相同的文件夹中。
・代码的执行如下所示。

  #命令行的情况
  python 脚本名

  #包括IPython的python控制台的情况如下所示。
  run 脚本名

・7-5-2以后的脚本执行需要时间。

[修改的地方]
・P183
  删除第2行的dt.
  错误：tmp['day'] = tmp.index.dt.day
  ↓
  正确：tmp['day'] = tmp.index.day

・P180
  在第21行fillna中追加inplace=True
  错误：tmp["sunhour"].fillna(-1)
  ↓
  正确：tmp["sunhour"].fillna(-1,inplace=True)

・P184
  最后，输出图7-8的代码不是7-5-3-2-graph.py，而是7-5-5-4-graph.py的错误。
  另外，因为在情节对象中包含了学习数据，所以图7-8也多少有些不同。
