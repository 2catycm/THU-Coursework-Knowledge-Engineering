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
train_dataset = Dataset.from_dict({
    "sentence1": ["What are pandas?", "What are pandas?"],
    "sentence2": ["Pandas are a kind of bear.", "Pandas are a kind of fish."],
    "label": [1.0, 0],
})
loss = losses.BinaryCrossEntropyLoss(model)

trainer = CrossEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()