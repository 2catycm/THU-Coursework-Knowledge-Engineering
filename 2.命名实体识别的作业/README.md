# Assignment 2-Named Entity Recognition
本次实验使用微软亚洲研究院标注的新闻领域数据集MSRANER （https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA）。其中共包含3个文件，train1.txt为训练数据，词与词之间采用空格分隔，每个词后用/给出该词的标签；test1.txt为未分词的测试集；testright1.txt给出了测试集对应的标签，结构与train1.txt相同。

使用BIO和BIOES标注模式分别对train1.txt和testright1.txt中的标签进行预处理，得到每个字对应的标签。

选择合适的特征，使用pycrfsuite和sklearn-crfsuite中的CRF模型进行训练和预测，使用nervaluate包（https://github.com/MantisAI/nervaluate）对得到的标注结果进行评估，报告其中4种评估架构的准确率、召回率和F1值。

比较BIO和BIOES标注模式对NER结果带来的影响。不允许抄袭；不能只提交结果文件，需要将程序代码一并提交；给出实验报告，代码要有注释。

## Description
- 下载数据放到`./data/raw`文件夹下
- 补充完成`preprocess.py`中BIO和BIOES两种预处理函数, 运行 `python -u preprocess.py`获得`./data/processed`预处理后文件，示例如下
```
中 B-NT
共 I-NT
中 I-NT
央 I-NT
致 O
中 B-NT
国 I-NT
致 I-NT
公 I-NT
党 I-NT
十 I-NT
一 I-NT
大 I-NT
的 O
贺 O
词 O

各 O
位 O
```
- 补充完成`data_util.py`中数据加载部分及`main.py`中模型定义部分
- 运行`python -u main.py`
- 训练时间大约10分钟，测试结果ent_type f1在84左右