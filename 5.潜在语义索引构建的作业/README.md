利用jieba对给定的文档进行分词，不使用开源工具包的情况下自行计算每个词的TF-TDF

利用numpy基于LSI原理，对Term-Document矩阵构造新的矩阵，即SVD奇异值分解

基于以下关键词查询，基于LSI原理进行搜索，输出包括：查询关键词，查询结果，根据LSI矩阵K的不同取值得到的查询准度：（K=10,20,30,40,50,100），算法要求有相应的注释

查询关键词如下：
  戈贝尔确诊新冠  
  美国从阿富汗撤军    
  魔兽世界怀旧服       
  欧洲杯推迟
  苹果折叠手机专利
  汤姆汉克斯确诊       
  特鲁多自我隔离 
  疫情防控思政大课
  意大利封闭全国  
  英国央行紧急降息

## Description
- 补充完成`util.py`中数据加载部分及`main.py`中数据加载及TF-IDF、相似度计算等部分
- 运行`python -u main.py`
- 可以更改`main.py` line 66 n的值更改查询结果汇报
- *本实现方案中汇报的指标同时包含查询top-n文档准确率以及文档分类的准确率，分别对应`util.py`中的classification函数和search_topn_for_each_key函数，请思考两者评估方案的异同并说明该方案是否合理*
- 需要提交整个代码文档以及对应的实验报告