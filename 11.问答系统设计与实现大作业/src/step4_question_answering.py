import os
import json
import logging
import shutil
from typing import List, Dict, Any, Tuple, Optional

import torch
import evaluate # Hugging Face Evaluate library for metrics
from datasets import Dataset, DatasetDict, load_from_disk, Features, Value, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator, # Appropriate when padding is done in preprocessing
    pipeline
)
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm # For progress bars during custom processing

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose a base model suitable for Question Answering
# Qwen2 models can be fine-tuned for QA, or use a classic BERT-based model.
# For Chinese, models like "bert-base-chinese", "hfl/chinese-roberta-wwm-ext" are common.
# Let's assume we continue with a Qwen2 variant or a well-known Chinese BERT.
# MODEL_FOR_QA_TYPE = "Qwen2" # Options: "Qwen2", "SBERT_as_BERT", "BERT_Chinese"
MODEL_FOR_QA_TYPE = "Qwen3_0.6B" 
MODEL_FOR_QA_TYPE = "timpal0l/mdeberta-v3-base-squad2" 

if MODEL_FOR_QA_TYPE == "Qwen2":
    MODEL_FOR_QA_NAME = "Qwen/Qwen2-0.5B" # Or your preferred Qwen2 model
    TRUST_REMOTE_CODE_QA = True
elif MODEL_FOR_QA_TYPE == "Qwen3_0.6B":
    MODEL_FOR_QA_NAME = "Qwen/Qwen3-0.6B"
    TRUST_REMOTE_CODE_QA = True
elif MODEL_FOR_QA_TYPE == "SBERT_as_BERT": # Using SBERT's underlying BERT for QA fine-tuning
    MODEL_FOR_QA_NAME = "DMetaSoul/sbert-chinese-general-v1"
    TRUST_REMOTE_CODE_QA = False
elif MODEL_FOR_QA_TYPE == "BERT_Chinese":
    MODEL_FOR_QA_NAME = "bert-base-chinese" # A common baseline
    TRUST_REMOTE_CODE_QA = False
elif MODEL_FOR_QA_TYPE == "timpal0l/mdeberta-v3-base-squad2":
    MODEL_FOR_QA_NAME = "timpal0l/mdeberta-v3-base-squad2"
    TRUST_REMOTE_CODE_QA = False
else:
    raise ValueError("Invalid MODEL_FOR_QA_TYPE")

logger.info(f"--- Using QA Model Type: {MODEL_FOR_QA_TYPE}, Name: {MODEL_FOR_QA_NAME} ---")

DATA_DIR = "./data/"
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json") # train.json from DuReader subset

# Path for the SQuAD-like formatted data (before tokenization)
SQUAD_FORMATTED_DATA_DIR = os.path.join(DATA_DIR, f"squad_formatted_qa_{MODEL_FOR_QA_TYPE.lower()}")
# Path for the fully processed (tokenized) dataset
PROCESSED_QA_DATA_DIR = os.path.join(DATA_DIR, f"processed_qa_tokenized_{MODEL_FOR_QA_TYPE.lower().replace('/', '_')}")

OUTPUT_DIR_QA = f"./qa_model_output_{MODEL_FOR_QA_TYPE.lower().replace('/', '_')}"

# Preprocessing & Training Hyperparameters
MAX_SEQ_LENGTH_QA = 384  # Max length for [CLS] Q [SEP] C [SEP]
DOC_STRIDE = 128         # Stride for overlapping context windows for long documents
TRAIN_TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

# BATCH_SIZE_QA = 8 # QA models can be memory intensive, adjust as needed
BATCH_SIZE_QA = 64 # QA models can be memory intensive, adjust as needed
NUM_EPOCHS_QA = 2 # Fine-tuning for QA often requires fewer epochs
LEARNING_RATE_QA = 3e-5 # Common learning rate for QA
WEIGHT_DECAY_QA = 0.01

# Flag to force reprocessing of data
FORCE_REPROCESS_SQUAD_FORMAT = False
FORCE_REPROCESS_TOKENIZED_DATA = False

# --- Helper Functions ---

def find_answer_char_indices(context: str, answer_text: str) -> List[int]:
    """Finds all start character indices of the answer_text in the context."""
    starts = []
    idx = context.find(answer_text)
    while idx != -1:
        starts.append(idx)
        idx = context.find(answer_text, idx + 1)
    return starts if starts else [-1] # Return [-1] if not found, SQuAD format expects a list

def convert_to_squad_format(items: List[Dict]) -> List[Dict]:
    """Converts data from train.json format to SQuAD-like format."""
    squad_formatted_items = []
    skipped_count = 0
    for i, item in enumerate(tqdm(items, desc="Converting to SQuAD format")):
        question = item.get("question")
        # Use 'answer_sentence' as the context. If it's a list, take the first.
        # For this task, answer_sentence is the top ranked sentence, usually one.
        context_list = item.get("answer_sentence", [])
        context = context_list[0] if isinstance(context_list, list) and context_list else \
                  context_list if isinstance(context_list, str) else ""

        answer_text_list = item.get("answer", []) # DuReader answer can be a list
        answer_text = answer_text_list[0] if isinstance(answer_text_list, list) and answer_text_list else \
                      answer_text_list if isinstance(answer_text_list, str) else None

        if not question or not context or not answer_text:
            skipped_count += 1
            continue

        # For SQuAD, 'answers' is a dict with 'text' (list) and 'answer_start' (list)
        # Our DuReader data usually has one answer span.
        answer_starts_char = find_answer_char_indices(context, answer_text)

        # If answer is not found in context, we might mark it as unanswerable by SQuAD convention
        # (empty lists or specific handling). For training, we usually need an answer.
        if answer_starts_char[0] == -1:
            # logger.warning(f"Answer '{answer_text}' not found in context for qid {item.get('qid', i)}. Skipping.")
            # For SQuAD v2, unanswerable questions have answers: {'text': [], 'answer_start': []}
            # For SQuAD v1 like training, we typically need an answer. Let's skip if not found for simplicity here.
            skipped_count += 1
            continue

        squad_formatted_items.append({
            "id": str(item.get("qid", f"gen_id_{i}")), # Ensure unique ID
            "title": str(item.get("pid", "default_title")), # Optional
            "context": context,
            "question": question,
            "answers": {"text": [answer_text] * len(answer_starts_char), "answer_start": answer_starts_char}
        })
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} items during SQuAD conversion due to missing fields or answer not in context.")
    return squad_formatted_items

# Global tokenizer (initialized in main)
tokenizer_qa = None

def preprocess_qa_function_for_map(examples):
    """
    Preprocesses question-context pairs for extractive QA.
    Handles tokenization, long contexts via sliding window (stride), and maps answers to token indices.
    (Adapted from Hugging Face SQuAD processing examples)
    """
    if tokenizer_qa is None:
        raise ValueError("Global tokenizer_qa is not initialized!")

    # Some questions are lists of strings, ensure they are flattened if so (though our data has single strings)
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers_squad = examples["answers"] # List of SQuAD-style answer dicts

    # Tokenize, with truncation and padding handled later by data collator if needed,
    # or handle padding here if using DefaultDataCollator. The tutorial uses padding="max_length".
    # `return_overflowing_tokens=True` and `stride` handle long contexts.
    tokenized_examples = tokenizer_qa(
        questions,
        contexts,
        max_length=MAX_SEQ_LENGTH_QA,
        truncation="only_second",  # Truncate context (the second sequence)
        stride=DOC_STRIDE,
        return_overflowing_tokens=True, # Create multiple features for long contexts
        return_offsets_mapping=True,    # Essential for mapping char answers to token answers
        padding="max_length"            # Pad to max_length
    )

    # offset_mapping gives char start/end for each token.
    # overflowing_tokens gives mapping from new feature to original example.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = [] # To map features back to original examples


    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = 0 
        if tokenizer_qa.cls_token_id is not None:
            try:
                cls_index = input_ids.index(tokenizer_qa.cls_token_id)
            except ValueError:
                logger.warning(f"CLS token ID {tokenizer_qa.cls_token_id} not found in input_ids for example. Defaulting to index 0.")
                cls_index = 0 # Default to 0 if CLS token somehow not found (should not happen)
        elif tokenizer_qa.bos_token_id is not None:
            try:
                cls_index = input_ids.index(tokenizer_qa.bos_token_id)
                logger.info(f"Using BOS token ID {tokenizer_qa.bos_token_id} at index {cls_index} as cls_index.")
            except ValueError:
                logger.warning(f"BOS token ID {tokenizer_qa.bos_token_id} not found in input_ids. Defaulting to index 0.")
                cls_index = 0
        else:
            logger.warning("Neither CLS token nor BOS token ID found in tokenizer. Defaulting to index 0 for unanswerable.")
            cls_index = 0
        # Get the original example index and its SQuAD-style answer
        original_example_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][original_example_index]) # Store original example ID
        answer = answers_squad[original_example_index] # answers_squad is a list of dicts

        # If no answer is given (e.g. SQuAD v2 unanswerable), label CLS token
        if not answer["answer_start"] or not answer["text"] or answer["answer_start"][0] == -1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue
        
        # We take the first answer if multiple are provided (SQuAD format allows this)
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # sequence_ids() tells us which part of the input is context (1) and question (0)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_token_start = idx
        if idx == len(sequence_ids): # No context tokens (should not happen with valid input)
             tokenized_examples["start_positions"].append(cls_index)
             tokenized_examples["end_positions"].append(cls_index)
             continue
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_token_end = idx - 1

        # If the answer is not fully inside the current context snippet, label CLS
        if not (offsets[context_token_start][0] <= start_char and offsets[context_token_end][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise it's the start and end token positions
            token_idx_start = context_token_start
            while token_idx_start <= context_token_end and offsets[token_idx_start][0] < start_char:
                token_idx_start += 1
            
            token_idx_end = context_token_end
            while token_idx_end >= context_token_start and offsets[token_idx_end][1] > end_char:
                token_idx_end -= 1

            # Check if answer is fully contained within the identified token span
            if offsets[token_idx_start][0] <= start_char and offsets[token_idx_end][1] >= end_char:
                tokenized_examples["start_positions"].append(token_idx_start)
                tokenized_examples["end_positions"].append(token_idx_end)
            else: # Answer span is valid but token mapping failed to perfectly align (e.g. partial token)
                  # This indicates an issue or a very tricky alignment. Label CLS for safety.
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                # logger.debug(f"Answer char span [{start_char},{end_char}] for example {examples['id'][original_example_index]} could not be perfectly mapped to token span. Offsets: Start cand: {offsets[token_idx_start]}, End cand: {offsets[token_idx_end]}. Labeling CLS.")


    return tokenized_examples

# For SQuAD style evaluation
squad_metric = evaluate.load("squad")

from transformers import EvalPrediction # Import if not already
from sklearn.metrics import accuracy_score # For simple token accuracy metrics
def compute_qa_metrics(eval_pred: EvalPrediction): # Now only takes EvalPrediction
    """
    Computes simpler metrics like start/end token accuracy from logits.
    Full SQuAD EM/F1 requires more complex post-processing, typically done
    after trainer.predict() with access to original examples and offset mappings.
    """
    # eval_pred.predictions contains (start_logits, end_logits)
    # eval_pred.label_ids contains (start_positions, end_positions)
    
    start_logits, end_logits = eval_pred.predictions
    start_true, end_true = eval_pred.label_ids

    # It's possible label_ids might be None if not provided or handled correctly by dataset.
    # However, for QA, Trainer should provide them if they are in the dataset.
    if start_true is None or end_true is None:
        logger.warning("True start/end positions not available in eval_pred.label_ids. Skipping accuracy metrics.")
        return {} # Return empty if no labels to compare against

    pred_starts = np.argmax(start_logits, axis=-1)
    pred_ends = np.argmax(end_logits, axis=-1)

    # Simple accuracy of predicting the exact start and end token indices
    start_acc = accuracy_score(start_true.flatten(), pred_starts.flatten())
    end_acc = accuracy_score(end_true.flatten(), pred_ends.flatten())
    
    # You can add other simple metrics if desired
    
    logger.info(f"In compute_qa_metrics: eval_start_token_acc: {start_acc:.4f}, eval_end_token_acc: {end_acc:.4f}")
    return {
        "eval_start_token_acc": start_acc,
        "eval_end_token_acc": end_acc,
    }


# --- Main Script ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_QA, exist_ok=True)

    # Initialize Tokenizer Globally
    logger.info(f"Initializing tokenizer for QA: {MODEL_FOR_QA_NAME}")
    tokenizer_qa = AutoTokenizer.from_pretrained(MODEL_FOR_QA_NAME, trust_remote_code=TRUST_REMOTE_CODE_QA)
    if tokenizer_qa.pad_token is None:
        if tokenizer_qa.eos_token is not None:
            tokenizer_qa.pad_token = tokenizer_qa.eos_token
            logger.info(f"Set QA tokenizer.pad_token to eos_token ('{tokenizer_qa.eos_token}')")
        else:
            tokenizer_qa.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Added new [PAD] token to QA tokenizer. Pad token ID: {tokenizer_qa.pad_token_id}")

    # --- Dataset Loading and Caching ---
    # Path for SQuAD-like formatted data (before tokenization, but after char span calculation)
    squad_formatted_train_file = os.path.join(SQUAD_FORMATTED_DATA_DIR, "train_squad_format.json")
    squad_formatted_eval_file = os.path.join(SQUAD_FORMATTED_DATA_DIR, "eval_squad_format.json")

    if FORCE_REPROCESS_SQUAD_FORMAT and os.path.exists(SQUAD_FORMATTED_DATA_DIR):
        logger.info(f"FORCE_REPROCESS_SQUAD_FORMAT is True. Removing {SQUAD_FORMATTED_DATA_DIR}")
        shutil.rmtree(SQUAD_FORMATTED_DATA_DIR)
    
    squad_train_items: List[Dict]
    squad_eval_items: List[Dict]

    if os.path.exists(squad_formatted_train_file) and os.path.exists(squad_formatted_eval_file):
        logger.info("Loading SQuAD-formatted data from disk...")
        with open(squad_formatted_train_file, 'r', encoding='utf-8') as f:
            squad_train_items = json.load(f)
        with open(squad_formatted_eval_file, 'r', encoding='utf-8') as f:
            squad_eval_items = json.load(f)
    else:
        logger.info("SQuAD-formatted data not found. Converting raw data...")
        with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
            all_train_json_items = [json.loads(line) for line in f]
        
        raw_train_items, raw_eval_items = train_test_split(
            all_train_json_items, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        logger.info(f"Split train.json: {len(raw_train_items)} for training, {len(raw_eval_items)} for evaluation.")

        squad_train_items = convert_to_squad_format(raw_train_items)
        squad_eval_items = convert_to_squad_format(raw_eval_items)

        os.makedirs(SQUAD_FORMATTED_DATA_DIR, exist_ok=True)
        with open(squad_formatted_train_file, 'w', encoding='utf-8') as f:
            json.dump(squad_train_items, f, ensure_ascii=False, indent=2)
        with open(squad_formatted_eval_file, 'w', encoding='utf-8') as f:
            json.dump(squad_eval_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved SQuAD-formatted data to {SQUAD_FORMATTED_DATA_DIR}")

    # Now, process tokenized data with caching
    if FORCE_REPROCESS_TOKENIZED_DATA and os.path.exists(PROCESSED_QA_DATA_DIR):
        logger.info(f"FORCE_REPROCESS_TOKENIZED_DATA is True. Removing {PROCESSED_QA_DATA_DIR}")
        shutil.rmtree(PROCESSED_QA_DATA_DIR)

    tokenized_squad_ds_dict: DatasetDict
    if os.path.exists(PROCESSED_QA_DATA_DIR):
        logger.info(f"Loading tokenized QA dataset from {PROCESSED_QA_DATA_DIR}...")
        tokenized_squad_ds_dict = DatasetDict.load_from_disk(PROCESSED_QA_DATA_DIR)
    else:
        logger.info("Tokenized QA dataset not found. Processing and caching...")
        # Create Hugging Face Dataset objects from the SQuAD-like items
        # Define features carefully for Dataset to preserve structure, esp. 'answers'
        squad_features = Features({
            'id': Value('string'),
            'title': Value('string'),
            'context': Value('string'),
            'question': Value('string'),
            'answers': Sequence({'text': Value('string'), 'answer_start': Value('int32')})
        })
        
        train_dataset_squad = Dataset.from_list(squad_train_items, features=squad_features)
        eval_dataset_squad = Dataset.from_list(squad_eval_items, features=squad_features)

        raw_squad_ds_dict = DatasetDict({'train': train_dataset_squad, 'eval': eval_dataset_squad})
        logger.info(f"Raw SQuAD-like DatasetDict created: {raw_squad_ds_dict}")

        tokenized_squad_ds_dict = raw_squad_ds_dict.map(
            preprocess_qa_function_for_map,
            batched=True,
            remove_columns=raw_squad_ds_dict["train"].column_names # Remove original text cols
        )
        logger.info(f"Tokenized SQuAD dataset created: {tokenized_squad_ds_dict}")
        logger.info(f"Saving tokenized QA dataset to {PROCESSED_QA_DATA_DIR}...")
        tokenized_squad_ds_dict.save_to_disk(PROCESSED_QA_DATA_DIR)

    # --- Model, Trainer, Training ---
    logger.info(f"Loading QA model: {MODEL_FOR_QA_NAME}")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_FOR_QA_NAME,
        trust_remote_code=TRUST_REMOTE_CODE_QA
    )
    if len(tokenizer_qa) > qa_model.config.vocab_size: # type: ignore
        logger.info(f"Resizing QA model token embeddings from {qa_model.config.vocab_size} to {len(tokenizer_qa)}") # type: ignore
        qa_model.resize_token_embeddings(len(tokenizer_qa))

    from bigmodelvis import Visualization
    Visualization(qa_model).structure_graph()
    if MODEL_FOR_QA_TYPE == "Qwen3_0.6B":
        print(f"Pad token ID sed by collator: {qa_model.tokenizer.pad_token_id}")
        qa_model.model.config.pad_token_id = qa_model.tokenizer.pad_token_id


    training_args_qa = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR_QA, "training_checkpoints"),
        eval_strategy="epoch", # Use eval_strategy for compatibility with newer Transformers
        save_strategy="epoch",
        learning_rate=LEARNING_RATE_QA,
        per_device_train_batch_size=BATCH_SIZE_QA,
        per_device_eval_batch_size=BATCH_SIZE_QA,
        num_train_epochs=NUM_EPOCHS_QA,
        weight_decay=WEIGHT_DECAY_QA,
        load_best_model_at_end=True,
        metric_for_best_model="loss", # Using loss as EM/F1 needs complex postprocessing in compute_metrics
        # For full SQuAD eval, you'd set metric_for_best_model to 'f1' or 'em'
        # after implementing proper postprocessing in compute_metrics.
        logging_steps=max(1, len(tokenized_squad_ds_dict["train"]) // (BATCH_SIZE_QA * 10)),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    data_collator_qa = DefaultDataCollator() # Padding is done in preprocess_qa_function_for_map

    trainer_qa = Trainer(
        model=qa_model,
        args=training_args_qa,
        train_dataset=tokenized_squad_ds_dict["train"],
        eval_dataset=tokenized_squad_ds_dict["eval"],
        tokenizer=tokenizer_qa, # Pass tokenizer for saving and potentially for some collator details
        data_collator=data_collator_qa,
        compute_metrics=compute_qa_metrics # Pass the simplified metric computer
    )

    logger.info("Starting QA model training...")
    trainer_qa.train()
    
    logger.info("Training complete. Evaluating final model...")
    final_eval_results = trainer_qa.evaluate()
    logger.info(f"Final evaluation results: {final_eval_results}")


    # --- Save Final Model and Demonstrate Inference ---
    FINAL_QA_MODEL_PATH = os.path.join(OUTPUT_DIR_QA, "final_model_qa_pipeline")
    logger.info(f"Saving final QA model for pipeline usage to {FINAL_QA_MODEL_PATH}...")
    trainer_qa.save_model(FINAL_QA_MODEL_PATH)
    # Tokenizer is saved by trainer.save_model if passed to Trainer constructor

    from collections import defaultdict
    import collections # For OrderedDict in SQuAD postprocessing
    


    logger.info("\n--- Demonstrating QA Inference with Hugging Face Pipeline ---")
    qa_device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline(
        "question-answering",
        model=FINAL_QA_MODEL_PATH,
        tokenizer=FINAL_QA_MODEL_PATH, # Ensure tokenizer is also there
        device=qa_device,
        trust_remote_code=TRUST_REMOTE_CODE_QA
    )
    

    # Use an example from your dev/eval set for more relevant inference demo
    if squad_eval_items and len(squad_eval_items) > 0:
        sample_qa_item = squad_eval_items[0]
        logger.info(f"Using sample for QA inference: Q: {sample_qa_item['question']}, C: {sample_qa_item['context'][:100]}...")
        try:
            prediction = qa_pipeline(question=sample_qa_item['question'], context=sample_qa_item['context'])
            logger.info(f"QA Pipeline Prediction: {prediction}")
            logger.info(f"Original Answer Info: {sample_qa_item['answers']}")
        except Exception as e:
            logger.error(f"Error during QA pipeline inference: {e}")
            # Fallback example if error with dev set item
            logger.info("Using fallback generic QA example...")
            prediction = qa_pipeline(question="中国的首都在哪里？", context="中国的首都是北京。")
            logger.info(f"Fallback QA Pipeline Prediction: {prediction}")

    else:
        logger.warning("No eval items to pick a sample for QA inference. Using generic example.")
        prediction = qa_pipeline(question="中国的首都在哪里？", context="中国的首都是北京。")
        logger.info(f"Generic QA Pipeline Prediction: {prediction}")


    logger.info(f"QA script finished. Check outputs in {OUTPUT_DIR_QA}")