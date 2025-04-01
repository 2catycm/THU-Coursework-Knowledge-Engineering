## Assignment 6-Sentiment Analysis
使用 ELMo 预训练语言模型获取词向量

利用卷积神经网络（CNN）或者循环神经网络（RNN）处理情感分类问题

数据集使用电影评论数据集SST-2，已划分为训练集（train.tsv）开发集（dev.tsv）和测试集（test.tsv）

预训练模型来自 ELMo 官方 AllenNLP 发布的基于 Pytorch 实现的版本，包括预训练权重（elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5）和配置文件（elmo_2x4096_512_2048cnn_2xhighway_options.json）

使用 Pytorch 中的提供的 CNN 和 RNN 模块构建相应的深度神经网络，以上文获取到的词向量作为输入进行训练；选择 CNN 的同学推荐使用 TextCNN，选择 RNN 的同学推荐使用 LSTM

在测试集上测试情感分类的准确率，以acc作为评价指标

### Description
- 代码框架使用CNN模型，需要按照要求补充`./model/cnn.py` 和 `data_util.py`中部分函数，补充完毕后运行`python -u main.py`即可
- 原代码默认抽取10000条训练集进行训练，训练时间约30min，accuracy=0.86
- 全量数据训练时间约5小时，accuracy=0.89
- 也可在已有代码基础上自行实现RNN模型
- 需要提交整个代码文档以及对应的实验报告