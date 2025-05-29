import os
import torch
import numpy as np
import functools
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel, # For SBERT if only extracting embeddings, but we'll try AutoModelForSequenceClassification
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from sklearn.metrics import accuracy_score
import shutil # For removing processed dataset directory if needed

# --- Configuration ---
# Choose the model type: "Qwen2" or "SBERT"
MODEL_TYPE_TO_RUN = "SBERT" # Or "Qwen2"

if MODEL_TYPE_TO_RUN == "Qwen2":
    MODEL_NAME = "Qwen/Qwen2-0.5B"
elif MODEL_TYPE_TO_RUN == "SBERT":
    MODEL_NAME = "DMetaSoul/sbert-chinese-general-v1"
else:
    raise ValueError("Invalid MODEL_TYPE_TO_RUN. Choose 'Qwen2' or 'SBERT'.")

DATA_DIR = "./data/"
TRAIN_FILE = os.path.join(DATA_DIR, "train_questions.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_questions.txt")

# Path for the processed (tokenized and label-encoded) dataset
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR, f"processed_question_classification_{MODEL_TYPE_TO_RUN.lower().replace('/', '_')}")

OUTPUT_DIR = f"./{MODEL_TYPE_TO_RUN.lower().replace('/', '_')}_question_classifier_output"

MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3 # Adjust as needed, SBERT might need fewer epochs for fine-tuning
LEARNING_RATE = 2e-5
DATALOADER_NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1


# Global mapping for fine labels
id2label_fine_global = {}
label2id_fine_global = {} # Will be populated after loading/processing data

# --- 1. Data Loading and Preprocessing ---

def load_and_parse_data(file_path):
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
    tokenized_inputs = tokenizer_obj(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH
    )
    tokenized_inputs['labels'] = examples['fine_label_str'] # Assumes 'fine_label_str' is already integer encoded
    return tokenized_inputs

# Initialize tokenizer first, as it's needed for PROCESSED_DATASET_PATH uniqueness if vocab differs
# For Qwen, trust_remote_code might be needed. For SBERT, usually not.
tokenizer_trust_remote_code = True if "qwen" in MODEL_NAME.lower() else False
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=tokenizer_trust_remote_code)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer for {MODEL_NAME} missing pad_token, set to eos_token.")
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        print(f"Tokenizer for {MODEL_NAME} missing pad_token and eos_token, set to unk_token.")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Tokenizer for {MODEL_NAME} missing pad_token, eos_token, and unk_token. Added new pad_token '[PAD]'.")


# --- Check for cached processed dataset ---
# To force reprocessing, delete the PROCESSED_DATASET_PATH directory
FORCE_REPROCESS_DATA = False # Set to True to ignore cache and reprocess
if FORCE_REPROCESS_DATA and os.path.exists(PROCESSED_DATASET_PATH):
    print(f"FORCE_REPROCESS_DATA is True. Removing existing processed dataset at {PROCESSED_DATASET_PATH}")
    shutil.rmtree(PROCESSED_DATASET_PATH)


if os.path.exists(PROCESSED_DATASET_PATH):
    print(f"Loading processed dataset from {PROCESSED_DATASET_PATH}...")
    tokenized_datasets = DatasetDict.load_from_disk(PROCESSED_DATASET_PATH)
    
    # Load label mappings from the features of the loaded dataset
    # This assumes the 'fine_label_str' column was ClassLabel encoded before saving
    if 'train' in tokenized_datasets and tokenized_datasets['train'].features:
        fine_label_feature_loaded = tokenized_datasets['train'].features['labels'] # The encoded column is now 'labels'
        if isinstance(fine_label_feature_loaded, ClassLabel):
            num_fine_labels = fine_label_feature_loaded.num_classes
            id2label_fine_global.update({i: label for i, label in enumerate(fine_label_feature_loaded.names)})
            label2id_fine_global.update({label: i for i, label in id2label_fine_global.items()})
            print(f"Loaded label mappings: {num_fine_labels} fine labels.")
        else:
            print("Warning: Could not automatically load label mappings from dataset features. Ensure they are set if model needs them.")
            # Fallback or error, as num_fine_labels is crucial
            # For simplicity, this example might fail here if mappings aren't found.
            # A robust solution would re-derive them if not stored/loaded properly.
            # However, if we save after class_encode, this should be fine.
            # The 'labels' column in tokenized_datasets IS the integer encoded version.
            # We need num_fine_labels and the mappings for the model and compute_metrics.
            # Let's re-derive from raw if not found, assuming raw_datasets is always available
            print("Re-deriving label information as it wasn't directly in loaded dataset features in expected ClassLabel format for 'labels'.")
            raw_train_data_dict_temp = load_and_parse_data(TRAIN_FILE) # Load raw again to get all labels
            all_fine_labels_for_encoding = sorted(list(set(raw_train_data_dict_temp["fine_label_str"])))
            num_fine_labels = len(all_fine_labels_for_encoding)
            id2label_fine_global.update({i: label for i, label in enumerate(all_fine_labels_for_encoding)})
            label2id_fine_global.update({label: i for i, label in id2label_fine_global.items()})


    else:
        # This case should ideally not happen if dataset was saved correctly
        raise ValueError("Loaded dataset is missing 'train' split or its features. Cannot infer label mappings.")

    print("Processed dataset loaded from disk.")

else:
    print("Processed dataset not found. Building and saving...")
    raw_train_data_dict = load_and_parse_data(TRAIN_FILE)
    raw_test_data_dict = load_and_parse_data(TEST_FILE)
    train_dataset_hf = Dataset.from_dict(raw_train_data_dict)
    test_dataset_hf = Dataset.from_dict(raw_test_data_dict)
    raw_datasets = DatasetDict({'train': train_dataset_hf, 'test': test_dataset_hf})
    print("Raw datasets loaded for processing.")

    # Encode fine-grained labels using datasets.ClassLabel
    raw_datasets = raw_datasets.class_encode_column("fine_label_str")
    fine_label_feature = raw_datasets['train'].features['fine_label_str']
    num_fine_labels = fine_label_feature.num_classes
    id2label_fine_global.update({i: label for i, label in enumerate(fine_label_feature.names)})
    label2id_fine_global.update({label: i for i, label in id2label_fine_global.items()})
    print(f"Number of unique fine-grained labels: {num_fine_labels}")

    print("Tokenizing datasets...")
    # Pass the initialized tokenizer to the map function
    _tokenize_and_align_labels_fn_partial = functools.partial(tokenize_and_align_labels_fn, tokenizer_obj=tokenizer)
    tokenized_datasets = raw_datasets.map(
        _tokenize_and_align_labels_fn_partial,
        batched=True,
        remove_columns=['text', 'fine_label_str'] # Original 'fine_label_str' (string) is removed
                                                 # The integer encoded version is now in 'fine_label_str' after class_encode
                                                 # and then copied to 'labels' by tokenize_and_align_labels_fn
    )
    # After map, the integer labels are in 'labels'. Let's ensure the features reflect this.
    # The actual ClassLabel object might be on the original column name after class_encode.
    # When saving, it saves the current state. So 'labels' will be an int column.
    # We need to ensure that when loading, `fine_label_feature_loaded` points to the correct feature
    # or we reconstruct id2label_fine_global & num_fine_labels from scratch if loading.

    print(f"Saving processed dataset to {PROCESSED_DATASET_PATH}...")
    tokenized_datasets.save_to_disk(PROCESSED_DATASET_PATH)
    print("Processed dataset built and saved.")

# print(f"Tokenized datasets structure: {tokenized_datasets}")
# print(f"Example tokenized train entry: {tokenized_datasets['train'][0]}")


# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 2. Model Loading and Training ---
model_trust_remote_code = True if "qwen" in MODEL_NAME.lower() else False
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_fine_labels,
    id2label=id2label_fine_global,
    label2id=label2id_fine_global,
    trust_remote_code=model_trust_remote_code,
    ignore_mismatched_sizes=True # If tokenizer was resized
)

if len(tokenizer) > model.config.vocab_size:
    print(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)} for {MODEL_NAME}")
    model.resize_token_embeddings(len(tokenizer))

def get_coarse_label_from_fine_str(fine_label_str):
    return fine_label_str.split('_')[0]

def compute_metrics(eval_pred):
    logits, true_fine_ids_flat = eval_pred
    predicted_fine_ids_flat = np.argmax(logits, axis=-1)
    fine_accuracy = accuracy_score(true_fine_ids_flat, predicted_fine_ids_flat)

    # Ensure id2label_fine_global is populated
    if not id2label_fine_global:
        # This can happen if loading from disk and feature extraction failed, though we try to prevent it.
        raise ValueError("id2label_fine_global is not populated. Cannot compute coarse accuracy.")

    true_fine_labels_str = [id2label_fine_global.get(id_, "UNK_LABEL") for id_ in true_fine_ids_flat]
    predicted_fine_labels_str = [id2label_fine_global.get(id_, "UNK_LABEL") for id_ in predicted_fine_ids_flat]
    
    true_coarse_labels_str = [get_coarse_label_from_fine_str(label) for label in true_fine_labels_str]
    predicted_coarse_labels_str = [get_coarse_label_from_fine_str(label) for label in predicted_fine_labels_str]
    coarse_accuracy = accuracy_score(true_coarse_labels_str, predicted_coarse_labels_str)

    return {
        "eval_fine_accuracy": fine_accuracy,
        "eval_coarse_accuracy": coarse_accuracy,
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=max(1, int(len(tokenized_datasets["train"]) // (BATCH_SIZE * 4))),
    load_best_model_at_end=True,
    metric_for_best_model="eval_fine_accuracy",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"\nStarting model training for {MODEL_NAME}...")
trainer.train()

# --- 3. Evaluation ---
print(f"\nEvaluating model {MODEL_NAME} on the test set...")
eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("\n--- Final Evaluation Results ---")
print(f"Model: {MODEL_NAME}")
print(f"Fine-grained Accuracy: {eval_results.get('eval_fine_accuracy'):.4f}")
print(f"Coarse-grained Accuracy: {eval_results.get('eval_coarse_accuracy'):.4f}")

# --- 4. Save Final Model and Demonstrate Pipeline ---
FINAL_MODEL_PIPELINE_PATH = os.path.join(OUTPUT_DIR, "final_model_for_pipeline")
print(f"\nSaving final model for pipeline usage to {FINAL_MODEL_PIPELINE_PATH}...")
trainer.save_model(FINAL_MODEL_PIPELINE_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PIPELINE_PATH)

print("\n--- Demonstrating Inference with Hugging Face Pipeline ---")
pipe_device = 0 if torch.cuda.is_available() else -1
classifier_pipeline = pipeline(
    "text-classification",
    model=FINAL_MODEL_PIPELINE_PATH,
    tokenizer=FINAL_MODEL_PIPELINE_PATH,
    device=pipe_device,
    trust_remote_code=model_trust_remote_code # Important for some models like Qwen
)
sample_questions = ["姚明是谁？", "中国的首都在哪里？", "长城的定义是什么？"]
print(f"\nPredicting with pipeline for {MODEL_NAME} model, sample questions:")
for question in sample_questions:
    try:
        result = classifier_pipeline(question)
        print(f"Question: {question} => Prediction: {result}")
    except Exception as e:
        print(f"Error during pipeline prediction for '{question}': {e}")

print(f"\nProcess complete for {MODEL_NAME}. Output and final model saved to {FINAL_MODEL_PIPELINE_PATH}")