from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
from datasets import Dataset

CROSS_ENCODER_MODEL_NAME = "Alibaba-NLP/gte-multilingual-reranker-base"
MAX_SEQ_LENGTH_CROSS_ENCODER = 512
TRUST_REMOTE_CODE_CROSS_ENCODER = True


model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, 
        num_labels=1, 
        max_length=MAX_SEQ_LENGTH_CROSS_ENCODER,
        trust_remote_code=TRUST_REMOTE_CODE_CROSS_ENCODER,
    )
# train_dataset = Dataset.from_dict({
#     "sentence1": ["What are pandas?", "What are pandas?"],
#     "sentence2": ["Pandas are a kind of bear.", "Pandas are a kind of fish."],
#     "label": [1.0, 0],
# })
# loss = losses.BinaryCrossEntropyLoss(model)

# trainer = CrossEncoderTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     loss=loss,
# )
# trainer.train()




# 

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from datasets import load_dataset

# Load a dataset with queries, positives, and negatives
# eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = [
    {
    "query": "[TYPE:OBJ] 《死亡日记(2009年上映美国电影) 》是一部什么类型的电影？",
    "positive": [
      "影片类型： 恐怖"
    ],
    "negative": [
    #   "讲述的是在北美的一个荒无人烟的孤岛上，一家人意外唤醒了死去了亲戚，于是展开了一场死人与活人的较量……",
    ]
  },
]

# Initialize the evaluator
reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=samples,
    name="ms-marco-dev",
    show_progress_bar=True,
)
results = reranking_evaluator(model)
'''
CrossEncoderRerankingEvaluator: Evaluating the model on the ms-marco-dev dataset:
Queries: 10047    Positives: Min 0.0, Mean 1.1, Max 5.0   Negatives: Min 1.0, Mean 7.1, Max 10.0
         Base  -> Reranked
MAP:     34.03 -> 62.36
MRR@10:  34.67 -> 62.96
NDCG@10: 49.05 -> 71.05
'''
print(reranking_evaluator.primary_metric)
# => ms-marco-dev_ndcg@10
print(results[reranking_evaluator.primary_metric])
# => 0.7104656857184184