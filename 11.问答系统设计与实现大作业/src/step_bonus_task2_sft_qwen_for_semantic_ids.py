import os
import json
import logging
import shutil
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig # Using SFTTrainer from TRL
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np

# --- 0. Configuration & Logging ---
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model & Data Paths ---
# Choose a Qwen Instruct model suitable for SFT
SFT_MODEL_TYPE = "Qwen3_for_SemID" # Unique name for this task
# MODEL_SFT_NAME = "Qwen/Qwen2-0.5B-Instruct-AWQ" # AWQ for smaller footprint
MODEL_SFT_NAME = "Qwen/Qwen3-0.6B"
TRUST_REMOTE_CODE_SFT = True


DATA_DIR = "./data"
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json") # Questions and ground truth PIDs
SEMANTIC_IDS_FILE = os.path.join(DATA_DIR, "semantic_identifiers.json") # Output from addon_task1

# Path for the SFT formatted and tokenized dataset
SFT_PROCESSED_DATA_DIR_SEMID = os.path.join(DATA_DIR, f"sft_processed_data_semid_{SFT_MODEL_TYPE.lower().replace('/', '_')}")
OUTPUT_DIR_SFT_SEMID = f"./sft_semid_model_output_{SFT_MODEL_TYPE.lower().replace('/', '_')}"

# Training Hyperparameters
MAX_SEQ_LENGTH_SFT_SEMID = 256  # Consider: question length + semantic ID length + prompt template
TRAIN_TEST_SPLIT_RATIO_SEMID = 0.1
RANDOM_SEED_SEMID = 42
BATCH_SIZE_SFT_SEMID = 8
NUM_EPOCHS_SFT_SEMID = 3 # SFT might need a few epochs
LEARNING_RATE_SFT_SEMID = 2e-5
GRAD_ACCUMULATION_STEPS_SEMID = 2
LOGGING_STEPS_SFT_SEMID = 20

# PEFT/LoRA (Optional but recommended for larger models if not doing full fine-tune)
USE_PEFT_SEMID = False # Set to True to use LoRA
LORA_R_SEMID = 8
LORA_ALPHA_SEMID = 16
LORA_DROPOUT_SEMID = 0.05

FORCE_REPROCESS_SFT_DATA_SEMID = False # Set to True to force reprocessing data

# --- Helper Functions ---

def load_semantic_id_map(filepath: str) -> Dict[str, str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pid_to_semid_map = json.load(f)
        logger.info(f"Loaded {len(pid_to_semid_map)} semantic identifiers from {filepath}")
        return pid_to_semid_map
    except FileNotFoundError:
        logger.error(f"Semantic identifiers file not found at {filepath}. This is required.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filepath}.")
        raise

def format_sft_semid_example(question: str, semantic_id_target: str, tokenizer: AutoTokenizer) -> Optional[str]:
    """
    Formats a single example for SFT to generate semantic identifiers.
    The model learns to predict the semantic_id_target.
    """
    system_message = "你是一个AI助手，你的任务是根据用户提出的问题，为其推荐最相关文档的语义标识符。语义标识符是一串由连字符'-'分隔的数字。"
    user_message_content = f"问题：\n{question}\n\n请为上述问题生成最相关文档的语义标识符。"
    
    # The target completion is the semantic ID string
    assistant_message_content = semantic_id_target

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content},
        {"role": "assistant", "content": assistant_message_content}
    ]
    
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False # False, as SFTTrainer expects the full text including completion
        )
        # SFTTrainer usually expects the text to end with EOS if it's not added by template
        # if not formatted_text.endswith(tokenizer.eos_token):
        #    formatted_text += tokenizer.eos_token
        return formatted_text
    except Exception as e:
        logger.warning(f"SFT SemID: Error applying chat template for {tokenizer.name_or_path}: {e}. Using generic fallback.")
        prompt_part = f"<s>[INST] 系统: {system_message}\n用户: {user_message_content} [/INST]\n"
        completion_part = assistant_message_content
        return prompt_part + completion_part + tokenizer.eos_token


def create_sft_semid_dataset(
    train_json_items: List[Dict], 
    pid_to_semid_map: Dict[str, str], 
    tokenizer: AutoTokenizer
) -> Dataset:
    """Creates a dataset for SFT, where each entry has a 'text' field."""
    formatted_texts = []
    items_with_semid = 0
    items_without_semid = 0

    for item in tqdm(train_json_items, desc="Formatting SFT data for Semantic ID generation"):
        question = item.get("question")
        pid = str(item.get("pid")) # Ground truth document ID
        
        if not question or not pid:
            continue

        semantic_id_target = pid_to_semid_map.get(pid)
        if semantic_id_target:
            formatted_string = format_sft_semid_example(question, semantic_id_target, tokenizer)
            if formatted_string:
                formatted_texts.append({
                    "text": formatted_string,
                    "id": str(item.get("qid", "N/A")), # For reference
                    "question_text": question, # Keep for eval prompt generation
                    "true_semantic_id": semantic_id_target # Keep for eval comparison
                })
                items_with_semid += 1
        else:
            items_without_semid +=1
            # logger.debug(f"PID {pid} for question '{question[:30]}...' not found in semantic_id_map. Skipping.")

    logger.info(f"Formatted {items_with_semid} SFT examples with semantic IDs.")
    if items_without_semid > 0:
        logger.warning(f"Skipped {items_without_semid} items because their PIDs were not in the semantic_id_map.")
    
    if not formatted_texts: # If no data was generated
        return Dataset.from_list([]) # Return empty dataset

    return Dataset.from_list(formatted_texts)

def compute_semid_generation_metrics(eval_predictions, eval_references_semid_str: List[str]):
    """
    Computes Exact Match and Prefix Match Accuracy for generated semantic IDs.
    eval_predictions: List of generated semantic ID strings.
    eval_references_semid_str: List of true semantic ID strings.
    """
    if len(eval_predictions) != len(eval_references_semid_str):
        logger.error("Mismatch in length of predictions and references for semantic ID metrics.")
        return {"semid_exact_match": 0.0, "semid_avg_prefix_match_ratio": 0.0}

    exact_matches = 0
    prefix_match_ratios = []

    for pred_id_str, true_id_str in zip(eval_predictions, eval_references_semid_str):
        pred_id_str = str(pred_id_str).strip() # Ensure string and strip
        true_id_str = str(true_id_str).strip()

        if pred_id_str == true_id_str:
            exact_matches += 1
        
        pred_parts = pred_id_str.split('-')
        true_parts = true_id_str.split('-')
        
        common_prefix_len = 0
        for i in range(min(len(pred_parts), len(true_parts))):
            if pred_parts[i] == true_parts[i]:
                common_prefix_len += 1
            else:
                break
        
        # Ratio of correctly predicted prefix length to true ID length
        # If true_parts is empty (should not happen if data is good), ratio is 0 or 1 if pred is also empty
        if not true_parts:
            prefix_match_ratios.append(1.0 if not pred_parts else 0.0)
        else:
            prefix_match_ratios.append(common_prefix_len / len(true_parts))

    em_accuracy = exact_matches / len(eval_references_semid_str) if eval_references_semid_str else 0.0
    avg_prefix_match = np.mean(prefix_match_ratios) if prefix_match_ratios else 0.0
    
    return {
        "semid_exact_match": em_accuracy,
        "semid_avg_prefix_match_ratio": avg_prefix_match
    }

    
THINK_END_TAG_STR = "</think>" # Qwen3 uses this string tag

def simple_parse_qwen_final_answer(full_generated_text: str) -> str:
    """
    Parses the Qwen model's output to extract the final answer after the </think> tag.
    Args:
        full_generated_text (str): The complete text generated by the model.
    Returns:
        str: The extracted final answer.
    """
    # Split by the </think> tag
    parts = full_generated_text.split(THINK_END_TAG_STR, 1)
    if len(parts) > 1:
        # Content after the first </think> tag is considered the final answer part
        final_answer_part = parts[1]
    else:
        # No </think> tag found in the completion, assume the whole completion is the answer
        final_answer_part = full_generated_text

    # Clean up leading/trailing whitespaces and newlines
    final_answer_lines = [line for line in final_answer_part.splitlines() if line.strip()]
    cleaned_answer = "\n".join(final_answer_lines).strip()
    
    return cleaned_answer

# --- Main Script ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_SFT_SEMID, exist_ok=True)
    
    # Define the final model path early
    FINAL_SFT_SEMID_MODEL_PATH = os.path.join(OUTPUT_DIR_SFT_SEMID, "final_sft_semid_model")

    # 1. Initialize Tokenizer (globally for helper functions)
    logger.info(f"Initializing tokenizer for SFT Semantic ID Generation: {MODEL_SFT_NAME}")
    tokenizer_semid = AutoTokenizer.from_pretrained(
        MODEL_SFT_NAME, 
        trust_remote_code=TRUST_REMOTE_CODE_SFT,
        padding_side='left' # Important for causal LMs if batching for generation
    )
    if tokenizer_semid.pad_token is None:
        if tokenizer_semid.eos_token is not None:
            tokenizer_semid.pad_token = tokenizer_semid.eos_token
        else:
            tokenizer_semid.add_special_tokens({'pad_token': '[PAD]'})
        logger.info(f"SFT SemID Tokenizer: pad_token set to '{tokenizer_semid.pad_token}' (ID: {tokenizer_semid.pad_token_id})")

    # 2. Load auxiliary data
    pid_to_semantic_id_map = load_semantic_id_map(SEMANTIC_IDS_FILE)

    # Check if final model already exists
    if os.path.exists(FINAL_SFT_SEMID_MODEL_PATH) and os.path.exists(os.path.join(FINAL_SFT_SEMID_MODEL_PATH, "config.json")):
        logger.info(f"Found existing trained model at {FINAL_SFT_SEMID_MODEL_PATH}. Skipping training and proceeding to evaluation.")
        skip_training = True
        # Still need to load evaluation dataset for metrics
        sft_semid_dataset_dict: DatasetDict
        if os.path.exists(SFT_PROCESSED_DATA_DIR_SEMID):
            logger.info(f"Loading SFT Semantic ID processed dataset from {SFT_PROCESSED_DATA_DIR_SEMID} for evaluation...")
            sft_semid_dataset_dict = DatasetDict.load_from_disk(SFT_PROCESSED_DATA_DIR_SEMID)
        else:
            logger.info("SFT Semantic ID Processed dataset not found. Creating for evaluation...")
            # ...existing dataset creation code...
            raw_train_json_items_list = []
            with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_train_json_items_list.append(json.loads(line))

            logger.info(f"Total items originally in {TRAIN_JSON_FILE}: {len(raw_train_json_items_list)}")
            if raw_train_json_items_list:
                original_pids_sample = [item.get('pid') for item in raw_train_json_items_list[:5]]
                logger.info(f"Sample PIDs (and their types) from {TRAIN_JSON_FILE}: "
                            f"{[(pid, type(pid).__name__) for pid in original_pids_sample]}")

            logger.info(f"Sample PIDs (keys) from semantic_id_map: {list(pid_to_semantic_id_map.keys())[:5]}")

            all_train_json_items = []
            for item in raw_train_json_items_list:
                pid_from_train_json = item.get("pid")
                if pid_from_train_json is not None:
                    if str(pid_from_train_json) in pid_to_semantic_id_map:
                        all_train_json_items.append(item)

            logger.info(f"Number of items from {TRAIN_JSON_FILE} found in semantic_id_map (after type conversion): {len(all_train_json_items)}")

            if not all_train_json_items:
                logger.error("CRITICAL: No items from train.json have corresponding semantic IDs after type conversion. "
                            "Cannot proceed with SFT training for semantic ID generation. ")

            raw_train_items, raw_eval_items = train_test_split(
                all_train_json_items, test_size=TRAIN_TEST_SPLIT_RATIO_SEMID, random_state=RANDOM_SEED_SEMID
            )
            logger.info(f"Split data: {len(raw_train_items)} for SFT training, {len(raw_eval_items)} for SFT evaluation.")

            train_sft_ds = create_sft_semid_dataset(raw_train_items, pid_to_semantic_id_map, tokenizer_semid)
            eval_sft_ds = create_sft_semid_dataset(raw_eval_items, pid_to_semantic_id_map, tokenizer_semid)
            
            sft_semid_dataset_dict = DatasetDict({'train': train_sft_ds, 'eval': eval_sft_ds})
            if not train_sft_ds or len(train_sft_ds) == 0:
                raise ValueError("No training data created for SFT. Check semantic ID mapping and input files.")

            logger.info(f"Saving SFT Semantic ID processed dataset to {SFT_PROCESSED_DATA_DIR_SEMID}...")
            sft_semid_dataset_dict.save_to_disk(SFT_PROCESSED_DATA_DIR_SEMID)
    else:
        logger.info("No existing trained model found. Proceeding with training...")
        skip_training = False

        # 3. Dataset Loading and Caching for SFT
        sft_semid_dataset_dict: DatasetDict
        if FORCE_REPROCESS_SFT_DATA_SEMID and os.path.exists(SFT_PROCESSED_DATA_DIR_SEMID):
            logger.info(f"FORCE_REPROCESS_SFT_DATA_SEMID is True. Removing {SFT_PROCESSED_DATA_DIR_SEMID}")
            shutil.rmtree(SFT_PROCESSED_DATA_DIR_SEMID)

        if os.path.exists(SFT_PROCESSED_DATA_DIR_SEMID):
            logger.info(f"Loading SFT Semantic ID processed dataset from {SFT_PROCESSED_DATA_DIR_SEMID}...")
            sft_semid_dataset_dict = DatasetDict.load_from_disk(SFT_PROCESSED_DATA_DIR_SEMID)
        else:
            logger.info("SFT Semantic ID Processed dataset not found. Creating and caching...")
            raw_train_json_items_list = []
            with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_train_json_items_list.append(json.loads(line))

                    logger.info(f"Total items originally in {TRAIN_JSON_FILE}: {len(raw_train_json_items_list)}")
                    if raw_train_json_items_list:
                        # 打印前几个PID的原始类型，帮助确认
                        original_pids_sample = [item.get('pid') for item in raw_train_json_items_list[:5]]
                        logger.info(f"Sample PIDs (and their types) from {TRAIN_JSON_FILE}: "
                                    f"{[(pid, type(pid).__name__) for pid in original_pids_sample]}")

                    logger.info(f"Sample PIDs (keys) from semantic_id_map: {list(pid_to_semantic_id_map.keys())[:5]}")

                    all_train_json_items = []
                    for item in raw_train_json_items_list:
                        pid_from_train_json = item.get("pid")
                        if pid_from_train_json is not None:
                            # *** 将 train.json 中的 pid 转换为字符串类型进行比较 ***
                            if str(pid_from_train_json) in pid_to_semantic_id_map:
                                all_train_json_items.append(item)
                        # else: # 可选：记录那些在train.json中就没有pid的条目
                            # logger.warning(f"Item in train.json missing 'pid': {item.get('qid', 'Unknown QID')}")

                    logger.info(f"Number of items from {TRAIN_JSON_FILE} found in semantic_id_map (after type conversion): {len(all_train_json_items)}")

                    if not all_train_json_items:
                        logger.error("CRITICAL: No items from train.json have corresponding semantic IDs after type conversion. "
                                    "Cannot proceed with SFT training for semantic ID generation. ")
                        # exit("Exiting due to no valid training data for SFT Semantic ID generation.")

            raw_train_items, raw_eval_items = train_test_split(
                all_train_json_items, test_size=TRAIN_TEST_SPLIT_RATIO_SEMID, random_state=RANDOM_SEED_SEMID
            )
            logger.info(f"Split data: {len(raw_train_items)} for SFT training, {len(raw_eval_items)} for SFT evaluation.")

            train_sft_ds = create_sft_semid_dataset(raw_train_items, pid_to_semantic_id_map, tokenizer_semid)
            eval_sft_ds = create_sft_semid_dataset(raw_eval_items, pid_to_semantic_id_map, tokenizer_semid)
            
            sft_semid_dataset_dict = DatasetDict({'train': train_sft_ds, 'eval': eval_sft_ds})
            if not train_sft_ds or len(train_sft_ds) == 0:
                 raise ValueError("No training data created for SFT. Check semantic ID mapping and input files.")

            logger.info(f"Saving SFT Semantic ID processed dataset to {SFT_PROCESSED_DATA_DIR_SEMID}...")
            sft_semid_dataset_dict.save_to_disk(SFT_PROCESSED_DATA_DIR_SEMID)

        logger.info(f"SFT Semantic ID Dataset loaded: {sft_semid_dataset_dict}")
        if not (sft_semid_dataset_dict['train'] and len(sft_semid_dataset_dict['train']) > 0):
            raise ValueError("Training dataset is empty after processing for SFT Semantic ID generation.")

        # 4. Model Loading and Training
        logger.info(f"Loading base model for SFT Semantic ID Generation: {MODEL_SFT_NAME}")
        model_sft_semid = AutoModelForCausalLM.from_pretrained(
            MODEL_SFT_NAME,
            trust_remote_code=TRUST_REMOTE_CODE_SFT,
            # device_map="auto" # For multi-GPU or auto placement
        )
        if len(tokenizer_semid) > model_sft_semid.config.vocab_size: # type: ignore
            logger.info(f"Resizing SFT SemID model token embeddings from {model_sft_semid.config.vocab_size} to {len(tokenizer_semid)}")
            model_sft_semid.resize_token_embeddings(len(tokenizer_semid))
        
        peft_config_semid = None
        if USE_PEFT_SEMID:
            from peft import LoraConfig
            if "qwen2" in MODEL_SFT_NAME.lower(): # Example target modules for Qwen2
                lora_target_modules_semid = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else: 
                lora_target_modules_semid = ["q_proj", "v_proj"] # Generic fallback
                logger.warning(f"PEFT target modules not specifically set for {MODEL_SFT_NAME}, using generic {lora_target_modules_semid}.")
            peft_config_semid = LoraConfig(
                r=LORA_R_SEMID, lora_alpha=LORA_ALPHA_SEMID, lora_dropout=LORA_DROPOUT_SEMID,
                target_modules=lora_target_modules_semid, bias="none", task_type="CAUSAL_LM",
            )
            logger.info("PEFT/LoRA enabled for SFT Semantic ID generation.")

        # 5. SFT Training Arguments
        WEIGHT_DECAY_SFT = 0.01
        DATALOADER_NUM_WORKERS = 4
        sft_semid_training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR_SFT_SEMID, "training_checkpoints"),
            num_train_epochs=NUM_EPOCHS_SFT_SEMID,
            per_device_train_batch_size=BATCH_SIZE_SFT_SEMID,
            # per_device_eval_batch_size=BATCH_SIZE_SFT_SEMID, # Used by SFTTrainer if eval_dataset is provided
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS_SEMID,
            learning_rate=LEARNING_RATE_SFT_SEMID,
            weight_decay=WEIGHT_DECAY_SFT,
            warmup_ratio=0.1,
            logging_steps=LOGGING_STEPS_SFT_SEMID,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            report_to="none",
            seed=RANDOM_SEED_SEMID,
            # SFTTrainer doesn't use compute_metrics during training by default for gen metrics
            # eval_strategy="steps", # Can be set if eval_dataset is provided to SFTTrainer
            # eval_steps=EVAL_STEPS_SFT_SEMID,
        )

        # 6. Initialize SFTTrainer and Train
        trainer_sft_semid = SFTTrainer(
            model=model_sft_semid,
            args=sft_semid_training_args,
            train_dataset=sft_semid_dataset_dict["train"],
            eval_dataset=sft_semid_dataset_dict.get("eval"), # For loss/perplexity eval during training
            # dataset_text_field="text", # The column with "prompt + completion"
            # tokenizer=tokenizer_semid,
            # max_seq_length=MAX_SEQ_LENGTH_SFT_SEMID,
            # peft_config=peft_config_semid if USE_PEFT_SEMID else None,
            # packing=True, # Set to True if you have many short sequences and want to pack them
        )
        
        logger.info("Starting SFT training for Semantic ID generation...")
        trainer_sft_semid.train()
        
        logger.info(f"SFT training for Semantic ID generation complete. Saving final model to {FINAL_SFT_SEMID_MODEL_PATH}")
        trainer_sft_semid.save_model(FINAL_SFT_SEMID_MODEL_PATH)
        tokenizer_semid.save_pretrained(FINAL_SFT_SEMID_MODEL_PATH)

    # 7. Evaluation (Post-Training) - Always run regardless of skip_training
    logger.info("\n--- Starting Post-Training Evaluation for Semantic ID Generation ---")
    eval_model_semid = AutoModelForCausalLM.from_pretrained(
        FINAL_SFT_SEMID_MODEL_PATH, 
        trust_remote_code=TRUST_REMOTE_CODE_SFT, 
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    eval_tokenizer_semid = AutoTokenizer.from_pretrained(
        FINAL_SFT_SEMID_MODEL_PATH,
        trust_remote_code=TRUST_REMOTE_CODE_SFT,
        padding_side='left'
    )
    if eval_tokenizer_semid.pad_token is None: # Ensure pad token for generation
        eval_tokenizer_semid.pad_token = eval_tokenizer_semid.eos_token

    generated_semids_text = []
    true_semids_text_for_eval = []
    
    # The eval dataset from SFTTrainer (sft_semid_dataset_dict["eval"]) contains 'text', 'id', 'question_text', 'true_semantic_id'
    eval_data_for_gen = sft_semid_dataset_dict["eval"]
    if not eval_data_for_gen or len(eval_data_for_gen) == 0:
        logger.warning("Evaluation dataset for Semantic ID generation is empty. Skipping post-training eval metrics.")
    else:
        logger.info(f"Generating Semantic IDs for {len(eval_data_for_gen)} dev examples...")
        for eval_example in tqdm(eval_data_for_gen, desc="Generating Semantic IDs for Eval"):
            question = eval_example["question_text"]
            true_semantic_id = eval_example["true_semantic_id"]

            system_message = "你是一个AI助手，你的任务是根据用户提出的问题，为其推荐最相关文档的语义标识符。语义标识符是一串由连字符'-'分隔的数字。"
            user_message_content = f"问题：\n{question}\n\n请为上述问题生成最相关文档的语义标识符。"
            messages_for_gen = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message_content}]
            
            try:
                generation_prompt = eval_tokenizer_semid.apply_chat_template(
                    messages_for_gen, tokenize=False, add_generation_prompt=True
                )
            except Exception as e_tmpl_eval_semid:
                logger.warning(f"Eval SemID: Error applying chat template: {e_tmpl_eval_semid}. Using generic.")
                generation_prompt = f"<s>[INST] 系统: {system_message}\n用户: {user_message_content} [/INST]\n"

            try:
                inputs = eval_tokenizer_semid(generation_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH_SFT_SEMID - 20).to(eval_model_semid.device)
                gen_kwargs_semid_eval = {
                    "max_new_tokens": 20, # Semantic IDs are usually short
                    "pad_token_id": eval_tokenizer_semid.pad_token_id if eval_tokenizer_semid.pad_token_id is not None else eval_tokenizer_semid.eos_token_id,
                    "eos_token_id": eval_tokenizer_semid.eos_token_id,
                    "do_sample": False # Deterministic output for eval
                }
                if isinstance(gen_kwargs_semid_eval["eos_token_id"], list):
                    gen_kwargs_semid_eval["eos_token_id"] = gen_kwargs_semid_eval["eos_token_id"][0]

                outputs = eval_model_semid.generate(**inputs, **gen_kwargs_semid_eval) # type: ignore
                
                input_token_len = inputs.input_ids.shape[1]
                generated_tokens_only = outputs[0][input_token_len:]
                generated_semid = eval_tokenizer_semid.decode(generated_tokens_only, skip_special_tokens=True).strip()
            except Exception as e_gen_semid:
                logger.error(f"Error during SemID generation for question '{question[:30]}...': {e_gen_semid}")
                generated_semid = ""

            generated_semids_text.append(generated_semid)
            true_semids_text_for_eval.append(true_semantic_id)

        if generated_semids_text:
            semid_metrics = compute_semid_generation_metrics(generated_semids_text, true_semids_text_for_eval)
            logger.info(f"Final Semantic ID Generation Metrics for {MODEL_SFT_NAME} (from {FINAL_SFT_SEMID_MODEL_PATH}):")
            for metric_name, value in semid_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.warning("No semantic IDs generated for evaluation.")


    # 8. Demonstrate Inference
    logger.info(f"\n--- Demonstrating SFT Model for Semantic ID Generation ---")
    if os.path.exists(FINAL_SFT_SEMID_MODEL_PATH):
        semid_gen_pipeline = pipeline(
            "text-generation",
            model=FINAL_SFT_SEMID_MODEL_PATH,
            tokenizer=FINAL_SFT_SEMID_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=TRUST_REMOTE_CODE_SFT
        )
        
        sample_question_for_semid = "研究生考试相关的文章有哪些？" # Example from assignment
        
        system_msg_inf = "你是一个AI助手，你的任务是根据用户提出的问题，为其推荐最相关文档的语义标识符。语义标识符是一串由连字符'-'分隔的数字。"
        user_msg_inf = f"问题：\n{sample_question_for_semid}\n\n请为上述问题生成最相关文档的语义标识符。"
        messages_inf = [{"role": "system", "content": system_msg_inf}, {"role": "user", "content": user_msg_inf}]
        
        pipeline_prompt_semid = semid_gen_pipeline.tokenizer.apply_chat_template(
            messages_inf, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"\nPipeline input prompt for SemID generation:\n{pipeline_prompt_semid}")
        
        try:
            pipeline_outputs_semid = semid_gen_pipeline(
                pipeline_prompt_semid, 
                max_new_tokens=20,
                pad_token_id=semid_gen_pipeline.tokenizer.pad_token_id if semid_gen_pipeline.tokenizer.pad_token_id is not None else semid_gen_pipeline.tokenizer.eos_token_id,
                eos_token_id=semid_gen_pipeline.tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=False
            )
            full_generated_text_semid = pipeline_outputs_semid[0]['generated_text']
            
            # Extract just the generated ID part
            generated_id_only = ""
            if full_generated_text_semid.startswith(pipeline_prompt_semid):
                generated_id_only = full_generated_text_semid[len(pipeline_prompt_semid):].strip()
            else: # Fallback if prompt not perfectly stripped (depends on pipeline behavior)
                 # For chat templates, find where assistant response starts.
                assistant_response_start = semid_gen_pipeline.tokenizer.apply_chat_template(
                    messages_inf + [{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=False
                )
                assistant_response_start = assistant_response_start.replace(semid_gen_pipeline.tokenizer.eos_token, "") # Remove potential EOS from empty assistant
                if full_generated_text_semid.startswith(assistant_response_start): #If assistant prompt part found
                     generated_id_only = full_generated_text_semid[len(assistant_response_start):].strip()
                else:
                    generated_id_only = full_generated_text_semid # As last resort

            logger.info(f"Sample Question for SemID: {sample_question_for_semid}")
            logger.info(f"Pipeline Generated Semantic ID: {generated_id_only}")
        except Exception as e_pipe_semid:
            logger.error(f"Error during SemID pipeline demonstration: {e_pipe_semid}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"Final SFT SemID model not found at {FINAL_SFT_SEMID_MODEL_PATH}. Skipping pipeline demo.")

    logger.info(f"SFT script for Semantic ID generation finished. Check outputs in {OUTPUT_DIR_SFT_SEMID}")