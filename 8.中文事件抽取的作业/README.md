# Event Extraction

获取词向量，可以使用中文预训练模型，也可以参照 DMCNN 中的方法自行训练 Skip-gram

进行事件触发词抽取，即从文本中抽取标识事件发生的触发词，触发词往往为动词和名词

进行事件论元抽取，即从文本中抽取触发词所对应的事件论元，论元主要为主体、客体、时间、地点，其中主体为必备论元

可以使用 Pipeline 模型也可以使用 Joint 模型

本次实验使用数据集来自2020科大讯飞事件抽取挑战赛初赛，数据为json格式，提供了文本以及相应的触发词、主体、客体、时间、地点，其中除触发词和主体以外，其他为可选字段；数据中给出的 distant_trigger 只是远程监督标签，并不是真实的标签，真实的触发词标签是 trigger；数据集已经分为了训练集、开发集和测试集，但由于是比赛数据，因此测试集没有提供标签，请同学们在开发集上进行评测

## 任务描述
- 识别文本中的事件触发词及其对应的论元
- 一句中可能包含多个事件
- 该实现识别事件触发词和不同类型的论元，但 *不* 对论元与事件进行配对

## 实现细节
- 整个流程由两阶段的序列标注构成
    - 第一阶段关注于事件的识别
    - 第二阶段负责论元的识别与分类
        - 在第二阶段，第一阶段识别出的事件触发词在测试阶段会被标记为 `<event> trigger <event/>`
- 默认采用中文 BERT `hfl/chinese-bert-wwm-ext` 作为基础模型

## 如何运行
###
- 补充 `preprocess.py` 和 `util.py` 中的 TODO 部分

### 数据准备
- 将数据放置在 `data/raw` 文件夹中
- 运行 `preprocess.py`，生成的结果分别存储在 `data/processed/argument` 和 `data/processed/trigger` 文件夹中，每个文件夹内包含 `train.txt` 和 `dev.txt`
    - 将原始的 `train.json` 和 `dev.json` 数据处理成字符标注格式
    - 由于一个事件触发词或论元可能跨越多个 token，因此标注格式采用 `IOB2`

### 训练
1. 执行 `python -u main.py --mode trigger` 运行第一阶段 
2. 执行 `python -u transform.py` 生成第二阶段的测试文件，运行后将在 `data/processed/argument/test.txt` 中看到结果。**注意：如果不使用 IOB2 格式，需要修改 `transform.py` 的部分内容**
3. 执行 `python -u main.py argument` 运行第二阶段
