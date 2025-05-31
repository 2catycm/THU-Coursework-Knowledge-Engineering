import os
import json
import logging
import shutil
from typing import List, Dict, Any, Tuple, Optional

import torch
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig # For 4-bit/8-bit quantization
)
from trl import SFTTrainer, SFTConfig # Using SFTTrainer from TRL
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import jieba
import os
import tempfile

FORCE_SFT_TRAINING = False # Set to True to always retrain
# FORCE_SFT_TRAINING = True # Set to True to always retrain

# Create a user-specific temp directory or a directory within your project
# For example, a .cache directory in your project's root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assuming script is in src
jieba_cache_dir = os.path.join(project_root, '.jieba_cache')
if not os.path.exists(jieba_cache_dir):
    os.makedirs(jieba_cache_dir)

jieba_cache_file = os.path.join(jieba_cache_dir, 'jieba.cache')
jieba.dt.cache_file = jieba_cache_file # Set the cache file path for the default tokenizer

from collections import Counter # For token counting

from rouge_chinese import Rouge # For Chinese ROUGE


# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose a base generative model. Instruct-tuned models are generally better for SFT.
# MODEL_SFT_TYPE = "Qwen2_0.5B_Instruct"
MODEL_SFT_TYPE = "Qwen3_0.6B" # Example, ensure this is a valid SFT target

MODEL_CONFIG_SFT = {
    "Qwen2": {"name": "Qwen/Qwen2-0.5B", "trust_remote_code": True},
    "SBERT": {"name": "DMetaSoul/sbert-chinese-general-v1", "trust_remote_code": False},
    "Qwen3_0.6B": {"name": "Qwen/Qwen3-0.6B", "trust_remote_code": True}, # Or Qwen/Qwen2-0.5B-Instruct etc.
    "RWKV": {"name": "fla-hub/rwkv7-168M-pile", "trust_remote_code": True} # Example RWKV, ensure compatibility
    ,"GTE": {"name": "Alibaba-NLP/gte-multilingual-reranker-base", "trust_remote_code": True} # Example RWKV, ensure compatibility
}


if MODEL_SFT_TYPE not in MODEL_CONFIG_SFT:
    raise ValueError(f"Unsupported MODEL_SFT_TYPE: {MODEL_SFT_TYPE}. Choose from {list(MODEL_CONFIG_SFT.keys())}")

MODEL_SFT_NAME = MODEL_CONFIG_SFT[MODEL_SFT_TYPE]["name"]
TRUST_REMOTE_CODE_SFT = MODEL_CONFIG_SFT[MODEL_SFT_TYPE]["trust_remote_code"]

DATA_DIR = "./data/"
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json")

# Path for the SFT formatted and tokenized dataset
# Cache path will depend on the model, as tokenization is model-specific
SFT_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, f"sft_processed_data_{MODEL_SFT_TYPE.lower().replace('/', '_')}")
OUTPUT_DIR_SFT = f"./sft_model_output_{MODEL_SFT_TYPE.lower().replace('/', '_')}"

# Training Hyperparameters
MAX_SEQ_LENGTH_SFT = 512  # Adjust based on typical prompt+answer length and model capacity
TRAIN_TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42
BATCH_SIZE_SFT = 4       # Generative models are VRAM hungry, start small
NUM_EPOCHS_SFT = 1       # SFT often requires fewer epochs
LEARNING_RATE_SFT = 1e-4 # Common for SFT
WEIGHT_DECAY_SFT = 0.01
GRAD_ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE_SFT * GRAD_ACCUMULATION_STEPS
LOGGING_STEPS_SFT = 10
EVAL_STEPS_SFT = 50      # Evaluate less frequently if eval is slow
SAVE_STEPS_SFT = 100
DATALOADER_NUM_WORKERS = 4

# PEFT/LoRA Configuration (Optional, SFTTrainer supports it easily)
USE_PEFT = False # Set to True to use LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# LORA_TARGET_MODULES will be model-specific, e.g. for Qwen2: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

FORCE_REPROCESS_SFT_DATA = False

# --- Helper Functions ---

def format_sft_train_example(question: str, context: str, answer: str, tokenizer: AutoTokenizer) -> Optional[str]:
    """
    Formats a single example into a string for SFT training.
    The model is trained to output the answer directly (no <think> block in the target completion).
    The prompt itself can be standard, and enable_thinking=False during template application for the assistant's turn
    or for the whole sequence if that's how the tokenizer expects it.
    """
    system_message = "你是一个乐于助人的问答助手。请根据提供的上下文和问题，简洁地回答问题。答案必须从上下文中获取或基于上下文推断。"
    user_message_content = f"上下文：\n{context}\n\n问题：\n{question}\n\n请根据以上信息回答。/no_think"
    
    # 训练时的目标完成就是直接的答案
    assistant_message_content = f"<think>\n\n</think>\n\n{answer}"

    messages_for_training = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content},
        {"role": "assistant", "content": assistant_message_content}
    ]
    
    try:
        # 对于训练数据，我们希望目标是直接的答案，不包含<think>...</think>。
        # enable_thinking=False (如果适用于整个模板应用) 确保模板本身不强制加入思考指令占位符。
        # SFTTrainer 会将最后一个 assistant message 作为学习目标。
        formatted_text = tokenizer.apply_chat_template(
            messages_for_training, 
            tokenize=False, 
            add_generation_prompt=False, # We are providing the full conversation for the 'text' field
            enable_thinking=False # Explicitly disable thinking structure in the template FOR TRAINING DATA
        )
        # SFTTrainer expects the 'text' field to be the full prompt + completion.
        # It will internally handle masking labels for the prompt part.
        # We add EOS token to signify end of sequence.
        return formatted_text #  SFTTrainer might add EOS if needed, or we can add tokenizer.eos_token
    except Exception as e:
        logger.warning(f"SFT Train: Error applying chat template for {tokenizer.name_or_path}: {e}. Using generic template.")
        # Generic fallback - ensure structure clearly separates prompt from target completion (answer)
        prompt_part = f"<s>[INST] 系统: {system_message}\n用户: {user_message_content} [/INST]\n"
        completion_part = assistant_message_content
        return prompt_part + completion_part + tokenizer.eos_token

def create_sft_dataset(items: List[Dict], tokenizer: AutoTokenizer) -> Dataset:
    """Creates a dataset for SFT where each entry has a 'text' field."""
    formatted_texts = []
    skipped_count = 0
    for item in tqdm(items, desc="Formatting SFT data"):
        question = item.get("question")
        context_list_or_str = item.get("answer_sentence", [])
        context = context_list_or_str[0] if isinstance(context_list_or_str, list) and context_list_or_str else \
                  context_list_or_str if isinstance(context_list_or_str, str) else ""
        
        answer_text_list_or_str = item.get("answer", [])
        answer_text = answer_text_list_or_str[0] if isinstance(answer_text_list_or_str, list) and answer_text_list_or_str else \
                      answer_text_list_or_str if isinstance(answer_text_list_or_str, str) else None

        if not question or not context or not answer_text:
            skipped_count += 1
            continue
        
        # Ensure answer_text is not empty string, as SFTTrainer might have issues.
        if not answer_text.strip():
            skipped_count +=1
            continue

        formatted_string = format_sft_train_example(question, context, answer_text, tokenizer)
        if formatted_string:
            formatted_texts.append({"text": formatted_string, 
                                    "id": str(item.get("qid", "N/A")), # Keep id for eval referencing
                                    "true_answer_text": answer_text # Keep original answer for eval
                                   })
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} items during SFT data formatting.")
    return Dataset.from_list(formatted_texts)

# For evaluation metrics
def tokenize_for_eval_metric(text: str) -> List[str]:
    if not text: return []
    return list(jieba.cut(text))

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

def parse_qwen_final_answer(full_generated_text: str, prompt_text_for_generation: str) -> str:
    """
    Parses the Qwen model's output to extract the final answer after the </think> tag.
    It also attempts to remove the input prompt if it's echoed in the generation.
    Args:
        full_generated_text (str): The complete text generated by the model, potentially including the prompt.
        prompt_text_for_generation (str): The prompt that was fed to the model for this generation.
    Returns:
        str: The extracted final answer.
    """
    # Step 1: Remove the input prompt from the generated text if it's present
    # This is important because model.generate() for causal LMs often includes the prompt.
    pure_completion = full_generated_text
    if full_generated_text.startswith(prompt_text_for_generation):
        pure_completion = full_generated_text[len(prompt_text_for_generation):]
    
    # Step 2: Split by the </think> tag
    parts = pure_completion.split(THINK_END_TAG_STR, 1)
    if len(parts) > 1:
        # Content after the first </think> tag is considered the final answer part
        final_answer_part = parts[1]
    else:
        # No </think> tag found in the completion, assume the whole completion is the answer
        final_answer_part = pure_completion

    # Clean up leading/trailing whitespaces and newlines
    # Qwen often has extra newlines after </think>
    final_answer_lines = [line for line in final_answer_part.splitlines() if line.strip()]
    cleaned_answer = "\n".join(final_answer_lines).strip()
    
    return cleaned_answer

def compute_generative_metrics(predictions: List[str], references_texts: List[List[str]], ids: List[str]):
    """
    Computes EM, F1 (token-based), BLEU, ROUGE.
    predictions: list of predicted answer strings
    references_texts: list of lists of true answer strings (each inner list for one example)
    ids: list of example ids corresponding to predictions and references
    """
    if len(predictions) != len(references_texts) or len(predictions) != len(ids):
        logger.error(f"Mismatch in lengths: predictions ({len(predictions)}), references ({len(references_texts)}), ids ({len(ids)})")
        return {}

    em_scores = []
    f1_scores = []
    
    # For SQuAD-like EM/F1
    squad_predictions = []
    squad_references = []

    for i in range(len(predictions)):
        pred_text = predictions[i]
        # For SQuAD eval, reference format needs 'answers': {'text': [...], 'answer_start': [-1]*len}
        # Here, we only have the text. 'id' is used for matching.
        # `answers` field for `evaluate`'s SQuAD metric needs `answer_start` which we don't have for generated text comparison.
        # So we calculate EM and token-F1 manually.
        
        true_ans_list_for_current_example = references_texts[i]
        
        # Exact Match (case/punctuation sensitive by default)
        current_em = 0
        if pred_text in true_ans_list_for_current_example:
            current_em = 1
        em_scores.append(current_em)

        # Token-based F1 (max over references)
        pred_tokens = tokenize_for_eval_metric(pred_text)
        max_f1_for_item = 0.0
        if not true_ans_list_for_current_example: # No ground truth
             max_f1_for_item = 1.0 if not pred_tokens else 0.0
        else:
            for true_ans_text in true_ans_list_for_current_example:
                true_tokens = tokenize_for_eval_metric(true_ans_text)
                
                if not true_tokens:
                    _f1 = 1.0 if not pred_tokens else 0.0
                elif not pred_tokens:
                    _f1 = 0.0
                else:
                    common_tokens = Counter(pred_tokens) & Counter(true_tokens)
                    num_common = sum(common_tokens.values())
                    precision = num_common / len(pred_tokens)
                    recall = num_common / len(true_tokens)
                    _f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                if _f1 > max_f1_for_item:
                    max_f1_for_item = _f1
        f1_scores.append(max_f1_for_item)

    # BLEU and ROUGE
    try:
        bleu_metric = evaluate.load("bleu")
        # predictions: list of strings, references: list of list of strings
        bleu_results = bleu_metric.compute(predictions=predictions, references=references_texts)
    except Exception as e:
        logger.error(f"BLEU computation error: {e}")
        bleu_results = {"bleu": 0.0}

    try:
        rouge_metric = evaluate.load("rouge") # Uses rouge_score, which can use jieba for Chinese
        # For Chinese ROUGE with evaluate, it might require pre-tokenized text (space separated words)
        # Or use rouge_chinese directly
        rouge_predictions_tokenized = [" ".join(tokenize_for_eval_metric(p)) for p in predictions]
        rouge_references_tokenized = [[" ".join(tokenize_for_eval_metric(r)) for r in ref_list] for ref_list in references_texts]
        
        # Using rouge_chinese directly for better Chinese handling
        rouge_calculator = Rouge()
        try:
            # rouge_chinese expects lists of space-separated tokens
            rouge_scores_all = rouge_calculator.get_scores(rouge_predictions_tokenized, [r[0] for r in rouge_references_tokenized], avg=True) # Use first ref for avg
            rouge_l_f = rouge_scores_all['rouge-l']['f']
        except ZeroDivisionError: # Can happen if all preds/refs are empty after tokenization
            logger.warning("ZeroDivisionError in ROUGE calculation (likely empty preds/refs). Setting ROUGE-L to 0.")
            rouge_l_f = 0.0
        except Exception as e_rouge:
            logger.error(f"Error calculating ROUGE with rouge_chinese: {e_rouge}")
            rouge_l_f = 0.0 # Fallback

        # Fallback or alternative using evaluate's rouge if rouge_chinese has issues or for comparison
        # hf_rouge_results = rouge_metric.compute(predictions=rouge_predictions_tokenized, references=rouge_references_tokenized)
        # rouge_l_f_hf = hf_rouge_results['rougeL']
    except Exception as e:
        logger.error(f"ROUGE computation error: {e}")
        # hf_rouge_results = {"rougeL": 0.0}
        rouge_l_f = 0.0


    return {
        "exact_match": np.mean(em_scores),
        "f1": np.mean(f1_scores),
        "bleu": bleu_results.get("bleu", 0.0),
        "rouge-l": rouge_l_f, # Using ROUGE-L F1 score
    }


# --- Main Script ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_SFT, exist_ok=True)

    # 1. Initialize Tokenizer (globally for helper functions)
    logger.info(f"Initializing tokenizer for SFT: {MODEL_SFT_NAME}")
    # For SFT, padding side is important for causal LMs if doing certain types of batching.
    # TRL's SFTTrainer usually handles this well. Let's use default from AutoTokenizer first.
    # Common practice for decoder-only models is tokenizer.padding_side = "left"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SFT_NAME, trust_remote_code=TRUST_REMOTE_CODE_SFT, padding_side='left')
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"SFT Tokenizer: Set pad_token to eos_token ('{tokenizer.eos_token}')")
        else: # Add a pad token if not available
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"SFT Tokenizer: Added new [PAD] token. Pad token ID: {tokenizer.pad_token_id}")
    
    # 2. Dataset Loading and Caching
    sft_dataset_dict: DatasetDict
    if FORCE_REPROCESS_SFT_DATA and os.path.exists(SFT_PROCESSED_DATA_DIR):
        logger.info(f"FORCE_REPROCESS_SFT_DATA is True. Removing {SFT_PROCESSED_DATA_DIR}")
        shutil.rmtree(SFT_PROCESSED_DATA_DIR)

    if os.path.exists(SFT_PROCESSED_DATA_DIR):
        logger.info(f"Loading SFT processed dataset from {SFT_PROCESSED_DATA_DIR}...")
        sft_dataset_dict = DatasetDict.load_from_disk(SFT_PROCESSED_DATA_DIR)
    else:
        logger.info("SFT Processed dataset not found. Creating and caching...")
        with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
            all_train_json_items = [json.loads(line) for line in f]
        
        raw_train_items, raw_eval_items = train_test_split(
            all_train_json_items, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        logger.info(f"Split train.json: {len(raw_train_items)} for SFT training, {len(raw_eval_items)} for SFT evaluation.")

        train_sft_ds = create_sft_dataset(raw_train_items, tokenizer)
        eval_sft_ds = create_sft_dataset(raw_eval_items, tokenizer)
        
        sft_dataset_dict = DatasetDict({'train': train_sft_ds, 'eval': eval_sft_ds})
        logger.info(f"Saving SFT processed dataset to {SFT_PROCESSED_DATA_DIR}...")
        sft_dataset_dict.save_to_disk(SFT_PROCESSED_DATA_DIR)

        logger.info(f"SFT Dataset loaded: {sft_dataset_dict}")
    if not (sft_dataset_dict['train'] and len(sft_dataset_dict['train']) > 0):
        raise ValueError("Training dataset is empty after processing. Check data and formatting.")

    # --- Define final model path ---
    FINAL_SFT_MODEL_PATH = os.path.join(OUTPUT_DIR_SFT, "final_sft_model")

    sft_model_for_evaluation: AutoModelForCausalLM 
    # Tokenizer for evaluation will be loaded from the saved model path or use the one from training.
    sft_tokenizer_for_evaluation: AutoTokenizer 

    # --- Conditional Training or Loading ---
    # Check for essential files like config.json and a model weights file
    model_config_file = os.path.join(FINAL_SFT_MODEL_PATH, "config.json")
    # Hugging Face models save weights as pytorch_model.bin, model.safetensors, or sharded versions
    model_weights_exist = (os.path.exists(os.path.join(FINAL_SFT_MODEL_PATH, "pytorch_model.bin")) or
                           os.path.exists(os.path.join(FINAL_SFT_MODEL_PATH, "model.safetensors")) or
                           # Check for sharded model index
                           os.path.exists(os.path.join(FINAL_SFT_MODEL_PATH, "pytorch_model.bin.index.json")) or
                           os.path.exists(os.path.join(FINAL_SFT_MODEL_PATH, "model.safetensors.index.json")))


    if not FORCE_SFT_TRAINING and os.path.exists(FINAL_SFT_MODEL_PATH) and model_weights_exist and os.path.exists(model_config_file):
        logger.info(f"Found existing fine-tuned SFT model at {FINAL_SFT_MODEL_PATH}. Skipping training.")
        logger.info("Loading model and tokenizer for evaluation...")
        sft_model_for_evaluation = AutoModelForCausalLM.from_pretrained(
            FINAL_SFT_MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE_SFT, # Consistent with how it would have been trained
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        sft_tokenizer_for_evaluation = AutoTokenizer.from_pretrained(
            FINAL_SFT_MODEL_PATH, # Load tokenizer from the same saved path
            trust_remote_code=TRUST_REMOTE_CODE_SFT,
            padding_side='left' # Ensure consistency
        )
        # Ensure pad token for the loaded tokenizer (important for generation)
        if sft_tokenizer_for_evaluation.pad_token is None:
            if sft_tokenizer_for_evaluation.eos_token is not None:
                sft_tokenizer_for_evaluation.pad_token = sft_tokenizer_for_evaluation.eos_token
            else:
                sft_tokenizer_for_evaluation.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Loaded evaluation tokenizer: pad_token set to '{sft_tokenizer_for_evaluation.pad_token}'")

    else:
        if FORCE_SFT_TRAINING:
            logger.info(f"FORCE_SFT_TRAINING is True. Proceeding with SFT training even if model exists.")
        else:
            logger.info(f"Fine-tuned SFT model not found at {FINAL_SFT_MODEL_PATH} or essential files missing. Proceeding with SFT training.")

        # 3. Load Base Model for SFT
        logger.info(f"Loading base model for SFT: {MODEL_SFT_NAME}")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_SFT_NAME,
            trust_remote_code=TRUST_REMOTE_CODE_SFT,
            # Optional: quantization_config=bnb_config if using BitsAndBytes
            # device_map="auto" # Can be useful for large models and PEFT
        )
        if len(tokenizer) > base_model.config.vocab_size: # type: ignore
            logger.info(f"Resizing SFT base model token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        
        # PEFT Configuration (if USE_PEFT is True)
        peft_config = None
        if USE_PEFT:
            # ... (your PEFT config logic as before) ...
            from peft import LoraConfig
            if "qwen2" in MODEL_SFT_NAME.lower():
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else: 
                lora_target_modules = ["q_proj", "v_proj"] 
                logger.warning(f"PEFT target modules not specifically set for {MODEL_SFT_NAME}, using generic {lora_target_modules}.")
            peft_config = LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                target_modules=lora_target_modules, bias="none", task_type="CAUSAL_LM",
            )
            logger.info("PEFT/LoRA enabled for SFT.")

        # 4. SFT Training Arguments
        sft_training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR_SFT, "training_checkpoints"),
            num_train_epochs=NUM_EPOCHS_SFT,
            per_device_train_batch_size=BATCH_SIZE_SFT,
            # per_device_eval_batch_size=BATCH_SIZE_SFT, # SFTTrainer uses eval_dataset for loss eval
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE_SFT,
            weight_decay=WEIGHT_DECAY_SFT,
            warmup_ratio=0.1,
            logging_steps=LOGGING_STEPS_SFT,
            # evaluation_strategy="steps", # SFTTrainer handles eval differently
            # eval_steps=EVAL_STEPS_SFT,   # if eval_dataset is provided to SFTTrainer
            save_strategy="epoch",
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            report_to="none",
            seed=RANDOM_SEED,
            # SFTTrainer doesn't use load_best_model_at_end directly in TrainingArguments for its main purpose.
            # It saves checkpoints based on save_strategy.
        )

        # 5. Initialize SFTTrainer
        trainer_sft = SFTTrainer(
            model=base_model, # Pass the base model here
            args=sft_training_args,
            train_dataset=sft_dataset_dict["train"],
            eval_dataset=sft_dataset_dict.get("eval"), # Optional, for eval loss during training
            # dataset_text_field="text",
            # tokenizer=tokenizer, # Pass the global tokenizer used for data prep
            # max_seq_length=MAX_SEQ_LENGTH_SFT,
            # peft_config=peft_config if USE_PEFT else None,
            # packing=True, # Consider if your sequences are short and you want to pack
        )
        
        logger.info("Starting SFT model training...")
        trainer_sft.train()
        
        logger.info(f"SFT training complete. Saving final model to {FINAL_SFT_MODEL_PATH}")
        trainer_sft.save_model(FINAL_SFT_MODEL_PATH) 
        # The tokenizer used for training should also be saved for consistency.
        # SFTTrainer/Trainer might do this, but explicit is good.
        tokenizer.save_pretrained(FINAL_SFT_MODEL_PATH)

        # For evaluation, load the model we just saved to ensure we're using the final state
        logger.info(f"Loading freshly trained model from {FINAL_SFT_MODEL_PATH} for evaluation.")
        sft_model_for_evaluation = AutoModelForCausalLM.from_pretrained(
            FINAL_SFT_MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE_SFT,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        sft_tokenizer_for_evaluation = AutoTokenizer.from_pretrained(
            FINAL_SFT_MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE_SFT,
            padding_side='left'
        )
        if sft_tokenizer_for_evaluation.pad_token is None: # Re-check for the loaded tokenizer
            if sft_tokenizer_for_evaluation.eos_token is not None:
                sft_tokenizer_for_evaluation.pad_token = sft_tokenizer_for_evaluation.eos_token
            else:
                sft_tokenizer_for_evaluation.add_special_tokens({'pad_token': '[PAD]'})


    # --- 7. Evaluation (Post-Training, Generative Metrics) ---
    if sft_model_for_evaluation and sft_tokenizer_for_evaluation:
        logger.info("\n--- Starting Post-Training Generative Evaluation ---")
        
        # Load raw_eval_items again or ensure it's available from the split
        if 'raw_eval_items' not in locals() or not raw_eval_items: # Check if it was defined in the 'else process_data' block
            logger.info("Loading raw eval items for final generative evaluation...")
            with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
                all_train_json_items_for_eval_final = [json.loads(line) for line in f]
            _, raw_eval_items = train_test_split(
                all_train_json_items_for_eval_final, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
            )

        generated_answers = []
        ground_truth_for_bleu_rouge = []
        eval_ids_final_list = [] # For SQuAD-like prediction list

        logger.info(f"Generating answers for {len(raw_eval_items)} dev examples using model: {FINAL_SFT_MODEL_PATH}...")
        for idx, eval_item in enumerate(tqdm(raw_eval_items, desc="Generating Eval Answers")):
            question = eval_item.get("question")
            context_list_or_str = eval_item.get("answer_sentence", [])
            context = context_list_or_str[0] if isinstance(context_list_or_str, list) and context_list_or_str else \
                      context_list_or_str if isinstance(context_list_or_str, str) else ""
            
            true_answer_text_list_or_str = eval_item.get("answer", [])
            true_answer = true_answer_text_list_or_str[0] if isinstance(true_answer_text_list_or_str, list) and true_answer_text_list_or_str else \
                          true_answer_text_list_or_str if isinstance(true_answer_text_list_or_str, str) else ""

            current_qid = str(eval_item.get("qid", f"eval_id_{idx}"))

            if not question or not context or not true_answer.strip():
                generated_answers.append("")
                ground_truth_for_bleu_rouge.append([""])
                eval_ids_final_list.append(current_qid)
                continue

            system_message = "你是一个乐于助人的问答助手。请根据提供的上下文和问题，简洁地回答问题。答案必须从上下文中获取或基于上下文推断。"
            user_message_content = f"上下文：\n{context}\n\n问题：\n{question}\n\n请根据以上信息回答。"
            messages_for_generation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message_content}
            ]
            
            try:
                generation_prompt_text = sft_tokenizer_for_evaluation.apply_chat_template(
                    messages_for_generation, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True # Allow thinking for evaluation
                )
            except Exception as e_tmpl_eval:
                logger.warning(f"Eval: Error applying chat template for gen: {e_tmpl_eval}. Using generic.")
                generation_prompt_text = f"<s>[INST] 系统: {system_message}\n用户: {user_message_content} [/INST]\n"
            
            try:
                inputs = sft_tokenizer_for_evaluation(generation_prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH_SFT - 100).to(sft_model_for_evaluation.device)
                gen_kwargs_eval = {
                    "max_new_tokens": 60,
                    "pad_token_id": sft_tokenizer_for_evaluation.pad_token_id if sft_tokenizer_for_evaluation.pad_token_id is not None else sft_tokenizer_for_evaluation.eos_token_id,
                    "eos_token_id": sft_tokenizer_for_evaluation.eos_token_id
                }
                if isinstance(gen_kwargs_eval["eos_token_id"], list): # Handle list of EOS tokens if any
                    gen_kwargs_eval["eos_token_id"] = gen_kwargs_eval["eos_token_id"][0]

                outputs = sft_model_for_evaluation.generate(**inputs, **gen_kwargs_eval)
                full_generated_text = sft_tokenizer_for_evaluation.decode(outputs[0], skip_special_tokens=False)
                predicted_final_answer = parse_qwen_final_answer(full_generated_text, generation_prompt_text)
            except Exception as e_gen_eval:
                logger.error(f"Error during generation for qid {current_qid}: {e_gen_eval}")
                predicted_final_answer = ""

            generated_answers.append(predicted_final_answer)
            ground_truth_for_bleu_rouge.append([simple_parse_qwen_final_answer(true_answer)])
            eval_ids_final_list.append(current_qid)
        
        # Compute metrics (ensure `predictions_for_squad_metric` is also populated if needed for your `compute_generative_metrics`)
        # The current `compute_generative_metrics` takes generated_answers, ground_truth_for_bleu_rouge, eval_ids_final_list
        final_metrics = compute_generative_metrics(generated_answers, ground_truth_for_bleu_rouge, eval_ids_final_list)
        logger.info(f"Final Generative Evaluation Metrics for {MODEL_SFT_NAME} (from {FINAL_SFT_MODEL_PATH}):")
        for metric_name, value in final_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    else:
        logger.error("SFT Model or Tokenizer for evaluation not loaded. Skipping final evaluation.")


    # --- 8. Save and Demonstrate Inference Pipeline ---
    # FINAL_SFT_MODEL_PATH already defined. Model should be saved there if training occurred.
    # If model was loaded, sft_model_for_evaluation and sft_tokenizer_for_evaluation hold them.
    
    logger.info(f"\n--- Demonstrating SFT Model Inference with Text Generation Pipeline ---")
    if os.path.exists(FINAL_SFT_MODEL_PATH): # Check again in case only eval was run
        pipe_device_sft = 0 if torch.cuda.is_available() else -1
        try:
            text_gen_pipeline = pipeline(
                "text-generation",
                model=FINAL_SFT_MODEL_PATH, # Load from the unified path
                tokenizer=FINAL_SFT_MODEL_PATH,
                # device=pipe_device_sft,
                trust_remote_code=TRUST_REMOTE_CODE_SFT # Use the config for the SFT model
            )
            # ... (pipeline inference demo as before, using parse_qwen_final_answer) ...
            sample_q_inf = "中国的首都在哪里？"
            sample_c_inf = "中华人民共和国的首都是北京，历史悠久，文化灿烂。"
            system_message_inf = "你是一个乐于助人的问答助手。请根据提供的上下文和问题，简洁地回答问题。答案必须从上下文中获取或基于上下文推断。"
            user_message_content_inf = f"上下文：\n{sample_c_inf}\n\n问题：\n{sample_q_inf}\n\n请根据以上信息回答。"
            messages_for_pipeline_inf = [{"role": "system", "content": system_message_inf}, {"role": "user", "content": user_message_content_inf}]
            
            pipeline_prompt_text_inf = text_gen_pipeline.tokenizer.apply_chat_template(
                messages_for_pipeline_inf, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            
            logger.info(f"\nPipeline input prompt (SFT):\n{pipeline_prompt_text_inf}")
            pipeline_outputs_inf = text_gen_pipeline(
                pipeline_prompt_text_inf, max_new_tokens=150, 
                pad_token_id=text_gen_pipeline.tokenizer.pad_token_id if text_gen_pipeline.tokenizer.pad_token_id is not None else text_gen_pipeline.tokenizer.eos_token_id,
                eos_token_id=text_gen_pipeline.tokenizer.eos_token_id,
                num_return_sequences=1)
            generated_full_text_pipeline_inf = pipeline_outputs_inf[0]['generated_text']
            
            pipeline_final_answer_inf = parse_qwen_final_answer(generated_full_text_pipeline_inf, pipeline_prompt_text_inf)
            logger.info(f"SFT Sample Question: {sample_q_inf}")
            logger.info(f"SFT Pipeline Extracted Final Answer: {pipeline_final_answer_inf}")
            logger.info(f"SFT Pipeline Answer: {generated_full_text_pipeline_inf}")

        except Exception as e_pipe_sft:
            logger.error(f"Error during SFT pipeline demonstration: {e_pipe_sft}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"Final SFT model not found at {FINAL_SFT_MODEL_PATH}. Skipping pipeline demonstration.")

    logger.info(f"SFT script finished. Check outputs in {OUTPUT_DIR_SFT}")
