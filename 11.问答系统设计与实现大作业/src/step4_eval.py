import os
import json
import torch
import evaluate # Hugging Face Evaluate library
from transformers import pipeline
from sklearn.model_selection import train_test_split # Only if regenerating dev set
from tqdm import tqdm # For progress bars
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (from your snippet, with slight cleanups for paths) ---
# MODEL_FOR_QA_TYPE_FROM_USER = "Qwen3_0.6B" # User's original, ensure mapping if it's a custom name
MODEL_FOR_QA_TYPE_FROM_USER = "timpal0l/mdeberta-v3-base-squad2" # User's current choice in snippet

# Clean up model name for directory creation (replace / with _)
MODEL_NAME_CLEAN_FOR_PATHS = MODEL_FOR_QA_TYPE_FROM_USER.replace('/', '_')

# Set TRUST_REMOTE_CODE based on model type (heuristic)
TRUST_REMOTE_CODE_QA = True if "qwen" in MODEL_FOR_QA_TYPE_FROM_USER.lower() else False

DATA_DIR = "./data/"
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json")

# Path for the SQuAD-like formatted data (this script will *read* the eval part)
SQUAD_FORMATTED_DATA_DIR = os.path.join(DATA_DIR, f"squad_formatted_qa_{MODEL_NAME_CLEAN_FOR_PATHS}")
EVAL_SQUAD_FORMAT_FILE = os.path.join(SQUAD_FORMATTED_DATA_DIR, "eval_squad_format.json")

# Path to the fine-tuned model (ensure this matches your training script's save path)
OUTPUT_DIR_QA_MODEL_TRAINING = f"./qa_model_output_{MODEL_NAME_CLEAN_FOR_PATHS}"
FINAL_QA_MODEL_PATH = os.path.join(OUTPUT_DIR_QA_MODEL_TRAINING, "final_model_qa_pipeline")

import jieba
import os
import tempfile

# Create a user-specific temp directory or a directory within your project
# For example, a .cache directory in your project's root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assuming script is in src
jieba_cache_dir = os.path.join(project_root, '.jieba_cache')
if not os.path.exists(jieba_cache_dir):
    os.makedirs(jieba_cache_dir)

jieba_cache_file = os.path.join(jieba_cache_dir, 'jieba.cache')
jieba.dt.cache_file = jieba_cache_file # Set the cache file path for the default tokenizer

from collections import Counter # For token counting

def tokenize_text_for_metric(text: str) -> List[str]:
    """Tokenizes text into words for metric calculation."""
    # For Chinese, Jieba is more appropriate than simple space splitting.
    # Ensure text is not None or empty before tokenizing
    if not text:
        return []
    return list(jieba.cut(text))

def compute_precision_recall_f1(prediction_tokens: List[str], ground_truth_tokens: List[str]) -> Tuple[float, float, float]:
    """
    Computes token-level precision, recall, and F1 score.
    """
    if not ground_truth_tokens: # If there's no ground truth, P/R/F1 are undefined or 0.
        return (1.0, 1.0, 1.0) if not prediction_tokens else (0.0, 0.0, 0.0)
    if not prediction_tokens: # If nothing is predicted, recall is 0. Precision is 1 if GT is also empty, else 0.
        return (1.0, 0.0, 0.0) if not ground_truth_tokens else (0.0, 0.0, 0.0)


    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common_tokens.values())

    precision = num_common / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
    recall = num_common / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
    
    f1 = 0.0
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return precision, recall, f1




# --- Helper function to find answer starts (if needed for re-generating eval_squad_format.json) ---
def find_answer_char_indices_eval(context: str, answer_text: str) -> List[int]:
    starts = []
    idx = context.find(answer_text)
    while idx != -1:
        starts.append(idx)
        idx = context.find(answer_text, idx + 1)
    return starts if starts else [-1] # SQuAD format expects a list; -1 if not found

def convert_to_squad_format_eval(items: List[Dict], desc="Converting to SQuAD format for eval") -> List[Dict]:
    squad_formatted_items = []
    skipped_count = 0
    for i, item in enumerate(tqdm(items, desc=desc)):
        question = item.get("question")
        # Context is usually answer_sentence from train.json
        context_list_or_str = item.get("answer_sentence", [])
        context = context_list_or_str[0] if isinstance(context_list_or_str, list) and context_list_or_str else \
                  context_list_or_str if isinstance(context_list_or_str, str) else ""

        # Answer text from train.json
        answer_text_list_or_str = item.get("answer", [])
        answer_text = answer_text_list_or_str[0] if isinstance(answer_text_list_or_str, list) and answer_text_list_or_str else \
                      answer_text_list_or_str if isinstance(answer_text_list_or_str, str) else None

        if not question or not context or not answer_text:
            # logger.debug(f"Skipping item (qid: {item.get('qid', 'N/A')}) due to missing q, c, or a.")
            skipped_count += 1
            continue

        answer_starts_char = find_answer_char_indices_eval(context, answer_text)
        
        # For SQuAD evaluation, we need valid answer starts.
        # If answer text is not found in context, this example can't be used for SQuAD EM/F1.
        if answer_starts_char[0] == -1:
            # logger.debug(f"Answer not found in context for qid {item.get('qid', 'N/A')}. Skipping.")
            skipped_count += 1
            continue

        squad_formatted_items.append({
            "id": str(item.get("qid", f"eval_id_{i}")),
            "title": str(item.get("pid", "eval_title")), # Optional for SQuAD
            "context": context,
            "question": question,
            "answers": {"text": [answer_text] * len(answer_starts_char), "answer_start": answer_starts_char}
        })
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} items during SQuAD conversion for eval (e.g., answer not in context).")
    return squad_formatted_items
# --- End Helper ---

def run_evaluation():
    logger.info(f"--- Starting Evaluation for Model: {FINAL_QA_MODEL_PATH} ---")

    # 1. Load SQuAD-formatted development (evaluation) data
    squad_eval_references: List[Dict] = []
    if os.path.exists(EVAL_SQUAD_FORMAT_FILE):
        logger.info(f"Loading SQuAD-formatted evaluation data from {EVAL_SQUAD_FORMAT_FILE}...")
        with open(EVAL_SQUAD_FORMAT_FILE, 'r', encoding='utf-8') as f:
            squad_eval_references = json.load(f)
    else:
        logger.warning(f"{EVAL_SQUAD_FORMAT_FILE} not found. Attempting to generate from {TRAIN_JSON_FILE}...")
        if not os.path.exists(TRAIN_JSON_FILE):
            logger.error(f"Error: {TRAIN_JSON_FILE} not found. Cannot generate or load evaluation data.")
            return
        with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
            all_train_json_items = [json.loads(line) for line in f]
        
        # Assuming the same split ratio and random seed as in training for consistency
        # This part should ideally match how your training script created its dev set
        RANDOM_SEED = 42 
        TRAIN_TEST_SPLIT_RATIO = 0.1 # Ensure this matches your training script's split
        _, raw_eval_items = train_test_split(
            all_train_json_items, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        squad_eval_references = convert_to_squad_format_eval(raw_eval_items)
        
        if squad_eval_references:
            os.makedirs(SQUAD_FORMATTED_DATA_DIR, exist_ok=True)
            with open(EVAL_SQUAD_FORMAT_FILE, 'w', encoding='utf-8') as f:
                json.dump(squad_eval_references, f, ensure_ascii=False, indent=2)
            logger.info(f"Generated and saved SQuAD-formatted eval data to {EVAL_SQUAD_FORMAT_FILE}")
        else:
            logger.error("Failed to generate SQuAD-formatted eval data.")
            return


    if not squad_eval_references:
        logger.error("No evaluation data loaded or generated. Exiting.")
        return

    # 2. Load QA Pipeline
    logger.info(f"Loading QA pipeline from {FINAL_QA_MODEL_PATH}...")
    qa_device = 0 if torch.cuda.is_available() else -1
    try:
        if not os.path.exists(FINAL_QA_MODEL_PATH):
            logger.error(f"Model path not found: {FINAL_QA_MODEL_PATH}. Ensure the model has been trained and saved correctly.")
            return
        qa_pipeline_instance = pipeline(
            "question-answering",
            model=FINAL_QA_MODEL_PATH,
            tokenizer=FINAL_QA_MODEL_PATH, # tokenizer should be saved with the model
            device=qa_device,
            trust_remote_code=TRUST_REMOTE_CODE_QA
        )
        logger.info("QA pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading QA pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Get Predictions
    predictions_for_squad_metric = []
    predictions_for_bleu_metric = [] 
    references_for_bleu_metric = []  

    logger.info(f"Generating predictions for {len(squad_eval_references)} evaluation examples...")
    for example in tqdm(squad_eval_references, desc="Evaluating"):
        question = example["question"]
        context = example["context"]
        example_id = example["id"]
        
        pipeline_output = qa_pipeline_instance(question=question, context=context)
        predicted_answer_text = pipeline_output.get('answer', "") # Default to empty if 'answer' key missing

        predictions_for_squad_metric.append({
            "id": example_id,
            "prediction_text": predicted_answer_text
            # SQuAD v2 also needs "no_answer_probability", but we are SQuAD v1 like here
        })
        
        predictions_for_bleu_metric.append(predicted_answer_text)
        # For BLEU, references should be list of lists of strings
        # example['answers']['text'] is already a list of strings (usually one for DuReader)
        references_for_bleu_metric.append(example['answers']['text'])


    # 4. Compute Metrics
    logger.info("\n--- Computing Metrics ---")

    logger.info("\n--- Computing Custom Precision, Recall, F1 ---")
    all_precisions = []
    all_recalls = []
    all_f1s = []

    if len(predictions_for_squad_metric) != len(squad_eval_references):
        logger.error("Mismatch between number of predictions and references. Skipping custom P/R/F1.")
    else:
        for pred_item, ref_item in zip(predictions_for_squad_metric, squad_eval_references):
            predicted_text = pred_item['prediction_text']
            true_answer_texts = ref_item['answers']['text'] # This is a list of true answers

            pred_tokens = tokenize_text_for_metric(predicted_text)
            
            max_f1_for_item = 0.0
            best_precision_for_item = 0.0
            best_recall_for_item = 0.0

            if not true_answer_texts: # No ground truth answers for this item
                if not pred_tokens: # Nothing predicted, nothing true
                    current_p, current_r, current_f1 = 1.0, 1.0, 1.0
                else: # Predicted something, but nothing true
                    current_p, current_r, current_f1 = 0.0, 0.0, 0.0
                best_precision_for_item = current_p
                best_recall_for_item = current_r
                max_f1_for_item = current_f1
            else:
                for true_ans_text in true_answer_texts:
                    true_tokens = tokenize_text_for_metric(true_ans_text)
                    precision, recall, f1 = compute_precision_recall_f1(pred_tokens, true_tokens)
                    if f1 > max_f1_for_item:
                        max_f1_for_item = f1
                        best_precision_for_item = precision
                        best_recall_for_item = recall
            
            all_precisions.append(best_precision_for_item)
            all_recalls.append(best_recall_for_item)
            all_f1s.append(max_f1_for_item)

        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        avg_f1 = np.mean(all_f1s) if all_f1s else 0.0

        logger.info(f"Custom Token-Overlap Metrics (averaged over examples, max F1 per example):")
        logger.info(f"  Average Precision: {avg_precision:.4f}")
        logger.info(f"  Average Recall:    {avg_recall:.4f}")
        logger.info(f"  Average F1 Score:  {avg_f1:.4f}")


    # --- Debug: Check references for 'title' key ---
    logger.info("Debug: Checking first 5 evaluation references for 'title' key...")
    # 去除 squad_eval_references 的 title context question 字段，保留answers和id
    for i, ref in enumerate(squad_eval_references):
        ref = {k: v for k, v in ref.items() if k in ['id', 'answers']}
        squad_eval_references[i] = ref
    # --- End Debug ---

    # SQuAD Metrics (EM, F1)
    try:
        squad_metric_calculator = evaluate.load("squad")
        # print(squad_metric_calculator)
        print(predictions_for_squad_metric[0])
        print(squad_eval_references[0])
        squad_results = squad_metric_calculator.compute(
            predictions=predictions_for_squad_metric, 
            references=squad_eval_references # This already has 'id' and 'answers' keys
        )
        logger.info(f"SQuAD Metrics:")
        logger.info(f"  Exact Match (EM): {squad_results.get('exact_match', 0.0):.4f}")
        logger.info(f"  F1 Score: {squad_results.get('f1', 0.0):.4f}")
    except Exception as e:
        logger.error(f"Error computing SQuAD metrics: {e}")
        import traceback
        traceback.print_exc()


    # BLEU Score
    try:
        bleu_metric_calculator = evaluate.load("bleu")
        # Ensure references_for_bleu_metric is List[List[str]]
        # Our current format is already correct if example['answers']['text'] is always a list.
        bleu_results = bleu_metric_calculator.compute(
            predictions=predictions_for_bleu_metric, 
            references=references_for_bleu_metric
        )
        logger.info(f"\nBLEU Score:")
        logger.info(f"  BLEU: {bleu_results.get('bleu',0.0):.4f}")
    except Exception as e:
        logger.error(f"Error computing BLEU score: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nNote: Precision and Recall for extractive QA are typically components of the SQuAD F1 score.")
    logger.info("--- Evaluation Complete ---")

if __name__ == "__main__":
    # Example: Manually set MODEL_FOR_QA_TYPE_FROM_USER if not taken from training script
    # MODEL_FOR_QA_TYPE_FROM_USER = "Qwen/Qwen2-0.5B" 
    # MODEL_NAME_CLEAN_FOR_PATHS = MODEL_FOR_QA_TYPE_FROM_USER.replace('/', '_')
    # TRUST_REMOTE_CODE_QA = "qwen" in MODEL_FOR_QA_TYPE_FROM_USER.lower()
    # OUTPUT_DIR_QA_MODEL_TRAINING = f"./qa_model_output_{MODEL_NAME_CLEAN_FOR_PATHS}"
    # FINAL_QA_MODEL_PATH = os.path.join(OUTPUT_DIR_QA_MODEL_TRAINING, "final_model_qa_pipeline")
    # SQUAD_FORMATTED_DATA_DIR = os.path.join(DATA_DIR, f"squad_formatted_qa_{MODEL_NAME_CLEAN_FOR_PATHS}")
    # EVAL_SQUAD_FORMAT_FILE = os.path.join(SQUAD_FORMATTED_DATA_DIR, "eval_squad_format.json")
    
    # Ensure the paths at the top match your trained model's output.
    # The snippet provided by you initializes these, so it should be fine.

    run_evaluation()