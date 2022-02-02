『Pythonによる機械学習入門 6章 画像による手形状分類』サンプルコードについて。
2016-11-28 ISP

[使用方法]
・将data.zip文件在与脚本文件相同的文件夹中解压缩。

・假设脚本从控制台执行。
 （仅在P140的HOG可视化时设定命令行）
   请注意必须指定参数。

・下面列出本书中的启动脚本。
P122
  run trial_handsign_SVM.py ./data/my_learn8/ ./data/my_test2/
  run trial_handsign_SVM.py ./data/my_learn10/ ./data/other_test2/

P126
  run classify_handsign_1.py 1 ./data/m01 ./data/m02 ./data/m03 ./data/m04
  run classify_handsign_1.py 1 ./data/other_test2 ./data/m02 ./data/m03 ./data/m04

P128
  run classify_handsign_1.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04 ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P130
  run classify_handsign_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04 ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P136
  run classify_handsign_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P140
  #这是执行命令行的情况。
  python viewHOG40.py ./data/m01/2/01_2_001.png

  # 如果是包含IPythhon的pythhon控制台，请执行以下操作。
  run viewHOG40.py ./data/m01/2/01_2_001.png

P142
  run classify_handsign_HOG_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P148
  run classify_handsign_HOG_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

[修正箇所]
・P124 リスト6-2
  60行目、61行目  print文の括弧を追加

・P129 リスト6-3
  60行目、61行目  print文の括弧を追加 


以上