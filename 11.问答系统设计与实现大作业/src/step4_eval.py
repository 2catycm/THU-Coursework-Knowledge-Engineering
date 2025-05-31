from transformers import pipeline
import os
import torch

# MODEL_FOR_QA_TYPE = "Qwen3_0.6B" 
MODEL_FOR_QA_TYPE = "timpal0l/mdeberta-v3-base-squad2" 
TRUST_REMOTE_CODE_QA = False

DATA_DIR = "./data/"
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json") # train.json from DuReader subset

# Path for the SQuAD-like formatted data (before tokenization)
SQUAD_FORMATTED_DATA_DIR = os.path.join(DATA_DIR, f"squad_formatted_qa_{MODEL_FOR_QA_TYPE.lower()}")
# Path for the fully processed (tokenized) dataset
PROCESSED_QA_DATA_DIR = os.path.join(DATA_DIR, f"processed_qa_tokenized_{MODEL_FOR_QA_TYPE.lower().replace('/', '_')}")

OUTPUT_DIR_QA = f"./qa_model_output_{MODEL_FOR_QA_TYPE.lower().replace('/', '_')}"

FINAL_QA_MODEL_PATH = os.path.join(OUTPUT_DIR_QA, "final_model_qa_pipeline")


qa_device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline(
    "question-answering",
    model=FINAL_QA_MODEL_PATH,
    tokenizer=FINAL_QA_MODEL_PATH, # Ensure tokenizer is also there
    device=qa_device,
    trust_remote_code=TRUST_REMOTE_CODE_QA
)
