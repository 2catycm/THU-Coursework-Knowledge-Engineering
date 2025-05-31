import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from sklearn.metrics import accuracy_score
import shutil
import functools

# --- Configuration ---
MODEL_TYPE_TO_RUN = "SBERT" # Or "Qwen2" - SET THIS TO THE MODEL CAUSING THE ERROR
BATCH_SIZE = 16
# MODEL_TYPE_TO_RUN = "RWKV"
MODEL_TYPE_TO_RUN = "Qwen3"
# BATCH_SIZE = 4
BATCH_SIZE = 16

if MODEL_TYPE_TO_RUN == "Qwen2":
    MODEL_NAME = "Qwen/Qwen2-0.5B"
    TOKENIZER_TRUST_REMOTE_CODE = True
    MODEL_TRUST_REMOTE_CODE = True
elif MODEL_TYPE_TO_RUN == "SBERT":
    MODEL_NAME = "DMetaSoul/sbert-chinese-general-v1"
    TOKENIZER_TRUST_REMOTE_CODE = False
    MODEL_TRUST_REMOTE_CODE = False # Usually False for BERT-based SBERT
elif MODEL_TYPE_TO_RUN == "Qwen3":
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    TOKENIZER_TRUST_REMOTE_CODE = True
    MODEL_TRUST_REMOTE_CODE = True
elif MODEL_TYPE_TO_RUN == "RWKV":
    MODEL_NAME = "fla-hub/rwkv7-168M-pile"
    TOKENIZER_TRUST_REMOTE_CODE = True
    MODEL_TRUST_REMOTE_CODE = True
else:
    raise ValueError("Invalid MODEL_TYPE_TO_RUN. Choose 'Qwen2' or 'SBERT'.")

print(f"--- Running for MODEL_TYPE: {MODEL_TYPE_TO_RUN}, MODEL_NAME: {MODEL_NAME} ---")

DATA_DIR = "./data/"
TRAIN_FILE = os.path.join(DATA_DIR, "train_questions.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_questions.txt")
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR, f"processed_qc_{MODEL_TYPE_TO_RUN.lower().replace('/', '_')}")
OUTPUT_DIR = f"./{MODEL_TYPE_TO_RUN.lower().replace('/', '_')}_qc_output_debug"

MAX_LENGTH = 128
# BATCH_SIZE = 16
# BATCH_SIZE = 4
# BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DATALOADER_NUM_WORKERS = 1 # Reduce for debugging, increase later if I/O is slow

id2label_fine_global = {}
label2id_fine_global = {}
num_fine_labels = 0 # Will be set

# --- 1. Data Loading and Preprocessing ---

def load_and_parse_data(file_path):
    # (Same as your existing function)
    labels = []
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                labels.append(parts[0])
                questions.append(parts[1])
            else:
                print(f"Warning: Skipping malformed line {line_num+1} in {file_path}: '{line}'")
    return {"fine_label_str": labels, "text": questions}

def tokenize_and_align_labels_fn(examples, tokenizer_obj):
    # (Same as your existing function)
    tokenized_inputs = tokenizer_obj(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH
    )
    tokenized_inputs['labels'] = examples['fine_label_str']
    return tokenized_inputs

print(f"Initializing tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TOKENIZER_TRUST_REMOTE_CODE)
print(f"Initial tokenizer: vocab_size={tokenizer.vocab_size}, pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}, eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}, unk_token='{tokenizer.unk_token}', unk_token_id={tokenizer.unk_token_id}")

ORIGINAL_TOKENIZER_LEN = len(tokenizer)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}).")
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        print(f"Set tokenizer.pad_token to unk_token ('{tokenizer.unk_token}', ID: {tokenizer.unk_token_id}).")
    else:
        print("Adding new [PAD] token to tokenizer.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Added new [PAD] token. New pad_token='{tokenizer.pad_token}', new pad_token_id={tokenizer.pad_token_id}.")
print(f"Final tokenizer: vocab_size (used by model for embedding size if resized)={len(tokenizer)}, pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")
if tokenizer.pad_token_id is None:
    raise ValueError("tokenizer.pad_token_id is None after setup! This will cause issues with padding.")


FORCE_REPROCESS_DATA = False # Set to True to force reprocessing
if FORCE_REPROCESS_DATA and os.path.exists(PROCESSED_DATASET_PATH):
    print(f"FORCE_REPROCESS_DATA is True. Removing existing processed dataset at {PROCESSED_DATASET_PATH}")
    shutil.rmtree(PROCESSED_DATASET_PATH)

if os.path.exists(PROCESSED_DATASET_PATH):
    print(f"Loading processed dataset from {PROCESSED_DATASET_PATH}...")
    tokenized_datasets = DatasetDict.load_from_disk(PROCESSED_DATASET_PATH)
    
    fine_label_feature_loaded = tokenized_datasets['train'].features['labels']
    if isinstance(fine_label_feature_loaded, ClassLabel):
        num_fine_labels = fine_label_feature_loaded.num_classes
        id2label_fine_global.update({i: label for i, label in enumerate(fine_label_feature_loaded.names)})
        label2id_fine_global.update({label: i for i, label in id2label_fine_global.items()})
        print(f"Loaded label mappings: {num_fine_labels} fine labels.")
    else:
        # Fallback if features are not perfectly restored (should not happen with save_to_disk)
        print("Warning: ClassLabel features not directly found in loaded dataset. Re-deriving label info...")
        temp_raw_train_data = load_and_parse_data(TRAIN_FILE)
        all_labels = sorted(list(set(temp_raw_train_data["fine_label_str"])))
        num_fine_labels = len(all_labels)
        id2label_fine_global.update({i: name for i, name in enumerate(all_labels)})
        label2id_fine_global.update({name: i for i, name in enumerate(all_labels)})
        print(f"Re-derived label mappings: {num_fine_labels} fine labels.")
else:
    print("Processed dataset not found. Building and saving...")
    raw_train_data_dict = load_and_parse_data(TRAIN_FILE)
    raw_test_data_dict = load_and_parse_data(TEST_FILE)
    train_dataset_hf = Dataset.from_dict(raw_train_data_dict)
    test_dataset_hf = Dataset.from_dict(raw_test_data_dict)
    raw_datasets = DatasetDict({'train': train_dataset_hf, 'test': test_dataset_hf})

    raw_datasets = raw_datasets.class_encode_column("fine_label_str")
    fine_label_feature = raw_datasets['train'].features['fine_label_str']
    num_fine_labels = fine_label_feature.num_classes
    id2label_fine_global.update({i: label for i, label in enumerate(fine_label_feature.names)})
    label2id_fine_global.update({label: i for i, label in id2label_fine_global.items()})

    _tokenize_fn_partial = functools.partial(tokenize_and_align_labels_fn, tokenizer_obj=tokenizer)
    tokenized_datasets = raw_datasets.map(
        _tokenize_fn_partial,
        batched=True,
        remove_columns=['text', 'fine_label_str'] 
    )
    # Ensure the 'labels' column (which originated from 'fine_label_str' ClassLabel) 
    # retains its ClassLabel features for the loaded dataset, if possible.
    # This happens automatically if the column being mapped to 'labels' was already ClassLabel.
    # Here, 'fine_label_str' was class_encoded, so its integer values are directly used for 'labels'.
    # The features for 'labels' in the new dataset will just be 'value'.
    # We save the original ClassLabel feature to restore it upon loading.
    tokenized_datasets.save_to_disk(PROCESSED_DATASET_PATH, 
    # PUSH_TO_HUB_THROTTLE_SECONDS=fine_label_feature
    )
     # type: ignore
    print(f"Processed dataset built and saved to {PROCESSED_DATASET_PATH}")

# --- Sanity check input_ids after tokenization ---
print("\n--- Sanity Checking input_ids ---")
max_id_overall = -1
for split in tokenized_datasets:
    max_id_split = -1
    for example_input_ids in tokenized_datasets[split]['input_ids']:
        if example_input_ids: # If not empty
            current_max = max(example_input_ids)
            if current_max > max_id_split:
                max_id_split = current_max
    print(f"Max token ID in '{split}' dataset: {max_id_split}")
    if max_id_split > max_id_overall:
        max_id_overall = max_id_split

print(f"Overall max token ID found in datasets: {max_id_overall}")
print(f"Current tokenizer vocabulary size (len(tokenizer)): {len(tokenizer)}")
if max_id_overall >= len(tokenizer):
    raise ValueError(f"Error: Max token ID ({max_id_overall}) is out of bounds for tokenizer vocab size ({len(tokenizer)}).")
print("Sanity check passed: All token IDs are within tokenizer vocab range.")
# --- End Sanity check ---

class DebuggingDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if 'labels' in batch:
            labels_tensor = batch['labels']
            print(f"DEBUG COLLATOR: Batch labels: min={labels_tensor.min().item() if labels_tensor.numel() > 0 else 'N/A'}, "
                  f"max={labels_tensor.max().item() if labels_tensor.numel() > 0 else 'N/A'}, "
                  f"shape={labels_tensor.shape}, num_fine_labels_config={num_fine_labels}")
            if labels_tensor.numel() > 0 and (labels_tensor.min() < 0 or labels_tensor.max() >= num_fine_labels):
                print(f"CRITICAL DEBUG COLLATOR: Label out of bounds! Min: {labels_tensor.min().item()}, Max: {labels_tensor.max().item()}. Expected range [0, {num_fine_labels-1}]")
        # You can also check input_ids here again if you suspect them despite earlier checks
        # if 'input_ids' in batch:
        #     input_ids_tensor = batch['input_ids']
        #     print(f"DEBUG COLLATOR: Batch input_ids: min={input_ids_tensor.min().item()}, max={input_ids_tensor.max().item()}")
        return batch

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# data_collator = DebuggingDataCollator(tokenizer=tokenizer)

print(f"\nLoading model {MODEL_NAME} for sequence classification...")
print(f"Number of fine labels for model: {num_fine_labels}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_fine_labels,
    id2label=id2label_fine_global,
    label2id=label2id_fine_global,
    trust_remote_code=MODEL_TRUST_REMOTE_CODE,
    ignore_mismatched_sizes=True # Important if resizing embeddings or changing classifier head
)
print(f"Model loaded. Initial model.config.vocab_size: {model.config.vocab_size}")

if len(tokenizer) > model.config.vocab_size:
    print(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)} for {MODEL_NAME}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model token embeddings resized. New model.config.vocab_size: {model.config.vocab_size}")
    print(f"Model input embedding matrix size: {model.get_input_embeddings().weight.size(0)}")
elif len(tokenizer) < model.config.vocab_size:
    print(f"Warning: Tokenizer length ({len(tokenizer)}) is less than model's vocab size ({model.config.vocab_size}). This is unusual if not intended.")
else:
    print(f"Tokenizer length ({len(tokenizer)}) matches model's vocab size ({model.config.vocab_size}). No resize needed.")

# Ensure the model's final embedding size matches len(tokenizer)
if model.get_input_embeddings().weight.size(0) != len(tokenizer):
    # This might indicate an issue if resize_token_embeddings didn't behave as expected
    # or if there's a mismatch not caught by the len(tokenizer) > model.config.vocab_size check
    print(f"CRITICAL WARNING: Model embedding size {model.get_input_embeddings().weight.size(0)} != len(tokenizer) {len(tokenizer)}")
    print("Attempting to resize again just in case, or this could be an issue with the model architecture or an earlier misconfiguration.")
    # model.resize_token_embeddings(len(tokenizer)) # Re-attempting resize could be risky if the state is already bad.
                                                 # Better to understand why there's a mismatch.
                                                 # For now, we will proceed, but this is a red flag.

def get_coarse_label_from_fine_str(fine_label_str): # (Same as before)
    return fine_label_str.split('_')[0]

def compute_metrics(eval_pred): # (Same as before, ensure id2label_fine_global is correctly populated)
    logits, true_fine_ids_flat = eval_pred
    predicted_fine_ids_flat = np.argmax(logits, axis=-1)
    fine_accuracy = accuracy_score(true_fine_ids_flat, predicted_fine_ids_flat)
    if not id2label_fine_global: raise ValueError("id2label_fine_global not populated for compute_metrics")
    true_fine_labels_str = [id2label_fine_global.get(id_, "UNK_LABEL") for id_ in true_fine_ids_flat]
    predicted_fine_labels_str = [id2label_fine_global.get(id_, "UNK_LABEL") for id_ in predicted_fine_ids_flat]
    true_coarse_labels_str = [get_coarse_label_from_fine_str(label) for label in true_fine_labels_str]
    predicted_coarse_labels_str = [get_coarse_label_from_fine_str(label) for label in predicted_fine_labels_str]
    coarse_accuracy = accuracy_score(true_coarse_labels_str, predicted_coarse_labels_str)
    return {"eval_fine_accuracy": fine_accuracy, "eval_coarse_accuracy": coarse_accuracy}

training_args = TrainingArguments( 
    # (Mostly same, ensure output_dir is unique if running multiple times)
    output_dir=os.path.join(OUTPUT_DIR, "training_checkpoints"), # Checkpoints in a subfolder
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=max(1, int(len(tokenized_datasets["train"]) // (BATCH_SIZE * 4 if BATCH_SIZE > 0 else 100))),
    load_best_model_at_end=True,
    metric_for_best_model="eval_fine_accuracy",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    report_to="none"
)

trainer = Trainer( # (Same as before)
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"\n--- Starting model training for {MODEL_NAME} ---")
print(f"Model embedding layer size: {model.get_input_embeddings().weight.size(0)}")
print(f"Tokenizer vocabulary size (len): {len(tokenizer)}")
print(f"Pad token ID used by collator: {data_collator.tokenizer.pad_token_id}")

try:
    model.config.pad_token_id = tokenizer.pad_token_id
except:
    pass

try:
    trainer.train()
except Exception as e:
    print(f"!!!!!! ERROR DURING TRAINING for {MODEL_NAME} !!!!!!")
    print(e)
    import traceback
    traceback.print_exc()
    # If it's a CUDA error, often helps to know the input IDs of the problematic batch
    # This is harder to get here directly, but the sanity check above should help.
    # You can also try running with CUDA_LAUNCH_BLOCKING=1 env variable for more precise error locations.
    # e.g. `CUDA_LAUNCH_BLOCKING=1 python your_script.py`
    raise # Re-raise the exception after printing diagnostics


print(f"\n--- Evaluating model {MODEL_NAME} on the test set ---")
eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("\n--- Final Evaluation Results ---")
print(f"Model: {MODEL_NAME}")
print(f"Fine-grained Accuracy: {eval_results.get('eval_fine_accuracy'):.4f}")
print(f"Coarse-grained Accuracy: {eval_results.get('eval_coarse_accuracy'):.4f}")

FINAL_MODEL_PIPELINE_PATH = os.path.join(OUTPUT_DIR, "final_model_for_pipeline")
print(f"\nSaving final model for pipeline usage to {FINAL_MODEL_PIPELINE_PATH}...")
trainer.save_model(FINAL_MODEL_PIPELINE_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PIPELINE_PATH)

# (Pipeline demonstration code can remain similar)
print("\n--- Demonstrating Inference with Hugging Face Pipeline ---")
pipe_device = 0 if torch.cuda.is_available() else -1
try:
    classifier_pipeline = pipeline(
        "text-classification",
        model=FINAL_MODEL_PIPELINE_PATH,
        tokenizer=FINAL_MODEL_PIPELINE_PATH,
        device=pipe_device,
        trust_remote_code=MODEL_TRUST_REMOTE_CODE
    )
    sample_questions = ["姚明是谁？", "中国的首都在哪里？", "长城的定义是什么？"]
    print(f"\nPredicting with pipeline for {MODEL_NAME} model, sample questions:")
    for question in sample_questions:
        result = classifier_pipeline(question)
        print(f"Question: {question} => Prediction: {result}")
except Exception as e:
    print(f"Error creating or using pipeline for {MODEL_NAME}: {e}")


print(f"\nProcess complete for {MODEL_NAME}. Output and final model saved to {FINAL_MODEL_PIPELINE_PATH}")