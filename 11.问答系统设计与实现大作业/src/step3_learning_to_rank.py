import os
import json
import random
import logging
from typing import List, Dict, Set, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset # Using Hugging Face datasets
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments, losses
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

from sentence_transformers.util import mine_hard_negatives
from sklearn.model_selection import train_test_split

from transformers import pipeline
import rwkv_custom_classifier # This will execute the registration
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
from datasets import Dataset

# Configure logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose the base model for the CrossEncoder
# Options: "Qwen2", "SBERT", "Qwen3_0.6B" (maps to a Qwen2 small variant), "RWKV"
# BASE_MODEL_TYPE = "SBERT" # Change this to experiment
# BASE_MODEL_TYPE = "RWKV" # Change this to experiment
BASE_MODEL_TYPE = "Qwen3_0.6B" # Change this to experiment
BASE_MODEL_TYPE = "GTE" # Change this to experiment

# Option to incorporate question type from Task 2
# USE_QUESTION_TYPE_FEATURE = True # Set to False to train without it for comparison
USE_QUESTION_TYPE_FEATURE = False

# Paths
DATA_DIR = "./data/"
PASSAGES_FILE = os.path.join(DATA_DIR, "passages_multi_sentences.json")
TRAIN_JSON_FILE = os.path.join(DATA_DIR, "train.json") # train.json from DuReader subset

# Path to your trained question classifier model (Task 2) - REPLACE WITH YOUR ACTUAL PATH
# This model will be loaded to predict question types if USE_QUESTION_TYPE_FEATURE is True
QUESTION_CLASSIFIER_MODEL_PATH = "./qwen3_qc_output_debug/final_model_for_pipeline" # Example path
# Or for SBERT based classifier if you preferred that:
# QUESTION_CLASSIFIER_MODEL_PATH = "./sbert_question_classifier_output/final_model_pipeline"

# Output directory for the trained sentence ranker
OUTPUT_DIR_RANKER = f"./sentence_ranker_output_{BASE_MODEL_TYPE}"
if USE_QUESTION_TYPE_FEATURE:
    OUTPUT_DIR_RANKER += "_with_qtype"

PROCESSED_DATA_DIR_NAME = "processed_ranking_data" # Subdirectory name for cached data
# The full path will be constructed in the main block using OUTPUT_DIR_RANKER

# Add a flag to force reprocessing if needed for debugging
FORCE_REPROCESS_RANKING_DATA = False 


# Model name mapping
MODEL_CONFIG = {
    "Qwen2": {"name": "Qwen/Qwen2-0.5B", "trust_remote_code": True},
    "SBERT": {"name": "DMetaSoul/sbert-chinese-general-v1", "trust_remote_code": False},
    "Qwen3_0.6B": {"name": "Qwen/Qwen3-0.6B", "trust_remote_code": True}, # Or Qwen/Qwen2-0.5B-Instruct etc.
    "RWKV": {"name": "fla-hub/rwkv7-168M-pile", "trust_remote_code": True} # Example RWKV, ensure compatibility
    ,"GTE": {"name": "Alibaba-NLP/gte-multilingual-reranker-base", "trust_remote_code": True} # Example RWKV, ensure compatibility

}

if BASE_MODEL_TYPE not in MODEL_CONFIG:
    raise ValueError(f"Unsupported BASE_MODEL_TYPE: {BASE_MODEL_TYPE}. Choose from {list(MODEL_CONFIG.keys())}")

CROSS_ENCODER_MODEL_NAME = MODEL_CONFIG[BASE_MODEL_TYPE]["name"]
TRUST_REMOTE_CODE_CROSS_ENCODER = MODEL_CONFIG[BASE_MODEL_TYPE]["trust_remote_code"]

# For Hard Negative Mining (if used) - choose a good bi-encoder
BI_ENCODER_FOR_MINING_NAME = "DMetaSoul/sbert-chinese-general-v1" # SBERT is good for this

# Training parameters
# NUM_EPOCHS = 5 
NUM_EPOCHS = 1 # CrossEncoders often fine-tune quickly, especially with good data
# TRAIN_BATCH_SIZE = 16 # Adjust based on GPU memory
# EVAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 64 # Adjust based on GPU memory
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5 # Common starting point for fine-tuning
WARMUP_STEPS_RATIO = 0.1
# MAX_SEQ_LENGTH_CROSS_ENCODER = 256 # Max length for [CLS] Q [SEP] S [SEP]
MAX_SEQ_LENGTH_CROSS_ENCODER = 1024 # Max length for [CLS] Q [SEP] S [SEP]
# MAX_SEQ_LENGTH_CROSS_ENCODER = 64 # Max length for [CLS] Q [SEP] S [SEP]

# --- Helper Functions ---

# Global question classifier pipeline (initialized if needed)
question_classifier_pipeline = None

def get_question_classifier():
    global question_classifier_pipeline
    if USE_QUESTION_TYPE_FEATURE and question_classifier_pipeline is None:
        if not os.path.exists(QUESTION_CLASSIFIER_MODEL_PATH):
            logger.warning(f"Question classifier model not found at {QUESTION_CLASSIFIER_MODEL_PATH}. Cannot use question type feature.")
            # Fallback or raise error - for now, disable feature if model not found
            globals()['USE_QUESTION_TYPE_FEATURE'] = False # Disable dynamically
            return None
        
        logger.info(f"Loading question classifier from: {QUESTION_CLASSIFIER_MODEL_PATH}")
        # Determine device for the pipeline
        device = 0 if torch.cuda.is_available() else -1
        # Trust remote code might be needed depending on the classifier model type (e.g., Qwen)
        # Assuming the classifier was Qwen based on example path
        classifier_trust_remote_code = "qwen" in QUESTION_CLASSIFIER_MODEL_PATH.lower()

        question_classifier_pipeline = pipeline(
            "text-classification",
            model=QUESTION_CLASSIFIER_MODEL_PATH,
            tokenizer=QUESTION_CLASSIFIER_MODEL_PATH, # Assuming tokenizer is saved with model
            device=device,
            trust_remote_code=classifier_trust_remote_code
        )
        logger.info("Question classifier loaded.")
    return question_classifier_pipeline


def get_question_type(question_text: str) -> str:
    """Predicts the question type using the loaded classifier."""
    if not USE_QUESTION_TYPE_FEATURE:
        return "" # Return empty string if feature is disabled
        
    classifier = get_question_classifier()
    if classifier is None: # If loading failed or feature got disabled
        return ""
        
    try:
        # Pipeline returns a list of dicts, e.g., [{'label': 'HUM_PERSON', 'score': 0.9...}]
        prediction = classifier(question_text)
        if prediction and isinstance(prediction, list) and prediction[0].get('label'):
            # We only care about the coarse type for the prompt for simplicity
            coarse_type = prediction[0]['label'].split('_')[0]
            return coarse_type
    except Exception as e:
        logger.error(f"Error predicting question type for '{question_text}': {e}")
    return "UNKNOWN_TYPE" # Fallback type


def load_passages(passages_file: str) -> Dict[str, List[str]]:
    """Loads passages from passages_multi_sentences.json into a dict {pid: [sentences]}."""
    passages_dict = {}
    with open(passages_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            passages_dict[str(data['pid'])] = data['document'] # Ensure pid is string
    logger.info(f"Loaded {len(passages_dict)} documents from {passages_file}")
    return passages_dict


def create_input_examples_for_cross_encoder( # Renamed back for consistency if you prefer, or keep _optimized
    source_json_data: List[Dict],
    all_passages: Dict[str, List[str]],
    question_to_type_map: Dict[str, str], # MODIFIED: Added parameter
    max_neg_per_positive: int = 5,
    include_q_type: bool = False # This flag is still useful to control behavior
) -> List[InputExample]:
    examples = []
    skipped_no_doc = 0
    skipped_no_ans_sent = 0

    # Classifier pipeline is NOT loaded here anymore. Assumed to be done once outside.

    for item_idx, item in enumerate(source_json_data):
        query_text_original = item['question']
        pid = str(item['pid'])
        answer_sentences_true = set(item.get('answer_sentence', []))

        if not query_text_original or not answer_sentences_true:
            skipped_no_ans_sent +=1
            continue

        document_sentences = all_passages.get(pid)
        if not document_sentences:
            skipped_no_doc +=1
            continue
        
        question_text_final = query_text_original
        if include_q_type:
            # Use the pre-computed map
            q_type = question_to_type_map.get(query_text_original, "UNKNOWN_TYPE") 
            if q_type and q_type != "UNKNOWN_TYPE": # Only add if type is known and valid
                question_text_final = f"[TYPE:{q_type}] {query_text_original}"
        
        # Add positive examples
        for ans_sent in answer_sentences_true:
            examples.append(InputExample(texts=[question_text_final, ans_sent], label=1.0))

        # Add negative examples
        negative_candidates = [s for s in document_sentences if s not in answer_sentences_true]
        num_to_sample = min(len(negative_candidates), max_neg_per_positive * len(answer_sentences_true))
        # Ensure num_to_sample is non-negative for random.sample
        if num_to_sample > 0:
            selected_negatives = random.sample(negative_candidates, num_to_sample)
            for neg_sent in selected_negatives:
                examples.append(InputExample(texts=[question_text_final, neg_sent], label=0.0))
        
        if (item_idx + 1) % 2000 == 0: # Adjusted logging frequency
            logger.info(f"Processed {item_idx+1}/{len(source_json_data)} items for InputExamples (current type: {'q_type' if include_q_type else 'no_q_type'})")
    
    # ... (rest of logging and return)
    logger.info(f"Created {len(examples)} InputExamples.")
    if skipped_no_doc > 0: logger.warning(f"Skipped {skipped_no_doc} items due to missing passages.")
    if skipped_no_ans_sent > 0: logger.warning(f"Skipped {skipped_no_ans_sent} items due to missing q/ans.")
    return examples


def create_reranking_evaluator_data(
    eval_json_data: List[Dict],
    all_passages: Dict[str, List[str]],
    question_to_type_map: Dict[str, str], # ADDED this parameter
    include_q_type: bool = False
) -> List[Dict]:
    evaluator_samples = []
    # Classifier pipeline is NOT loaded here anymore.

    for item_idx, item in enumerate(eval_json_data): # Added item_idx for potential logging
        query_text_original = item['question']
        pid = str(item['pid'])
        positive_sentences = item.get('answer_sentence', [])

        if not query_text_original or not positive_sentences:
            # logger.debug(f"Skipping item in dev set due to missing query or positive_sentences: {item.get('qid', 'N/A')}")
            continue

        document_sentences = all_passages.get(pid)
        if not document_sentences:
            # logger.debug(f"Skipping item in dev set due to missing document for pid {pid}: {item.get('qid', 'N/A')}")
            continue

        question_text_final = query_text_original
        if include_q_type:
            # Use the pre-computed map
            q_type = question_to_type_map.get(query_text_original, "UNKNOWN_TYPE") 
            if q_type and q_type != "UNKNOWN_TYPE": # Only add if type is known and valid
                question_text_final = f"[TYPE:{q_type}] {query_text_original}"

        negative_sentences = [s for s in document_sentences if s not in positive_sentences]

        # The evaluator needs at least one positive sentence.
        # Negative sentences can be empty.
        evaluator_samples.append({
            'query': question_text_final,
            'positive': positive_sentences,
            'negative': negative_sentences
        })

        # Optional: add logging for processing dev evaluator data if it's slow or you want to track it
        # if (item_idx + 1) % 500 == 0:
        #     logger.info(f"Processed {item_idx+1}/{len(eval_json_data)} items for Reranking Evaluator data")

    logger.info(f"Created {len(evaluator_samples)} samples for CrossEncoderRerankingEvaluator.")
    return evaluator_samples


# --- Main Script ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_RANKER, exist_ok=True)

    # Define the specific path for processed data based on OUTPUT_DIR_RANKER
    processed_data_path_specific = os.path.join(OUTPUT_DIR_RANKER, PROCESSED_DATA_DIR_NAME)

    train_examples: List[InputExample]
    dev_evaluator_data: List[Dict]

    if FORCE_REPROCESS_RANKING_DATA and os.path.exists(processed_data_path_specific):
        logger.info(f"FORCE_REPROCESS_RANKING_DATA is True. Removing existing processed ranking data at {processed_data_path_specific}")
        shutil.rmtree(processed_data_path_specific) # Ensure shutil is imported

    if os.path.exists(processed_data_path_specific):
        logger.info(f"Attempting to load pre-processed ranking data from {processed_data_path_specific}...")
        try:
            with open(os.path.join(processed_data_path_specific, "cached_train_examples.json"), 'r', encoding='utf-8') as f:
                train_examples_json = json.load(f)
                train_examples = [InputExample(texts=ex['texts'], label=ex['label']) for ex in train_examples_json]
            with open(os.path.join(processed_data_path_specific, "cached_dev_evaluator_data.json"), 'r', encoding='utf-8') as f:
                dev_evaluator_data = json.load(f)
            logger.info("Successfully loaded cached training examples and dev evaluator data.")
        except FileNotFoundError:
            logger.warning(f"Cache files not found in {processed_data_path_specific}, will reprocess.")
            # Fall through to reprocessing block
            process_and_cache_data = True
        except Exception as e:
            logger.error(f"Error loading cached data from {processed_data_path_specific}: {e}. Reprocessing data.")
            process_and_cache_data = True
        else:
            process_and_cache_data = False # Successfully loaded
    else:
        logger.info(f"Pre-processed ranking dataset not found at {processed_data_path_specific}. Will process and cache data.")
        process_and_cache_data = True

    if process_and_cache_data:
        logger.info("Processing data for ranking task...")
        passages = load_passages(PASSAGES_FILE)
        with open(TRAIN_JSON_FILE, 'r', encoding='utf-8') as f:
            all_train_json_items = [json.loads(line) for line in f]
        
        train_items_raw, dev_items_raw = train_test_split(all_train_json_items, test_size=0.1, random_state=42)
        logger.info(f"Split train.json: {len(train_items_raw)} for training, {len(dev_items_raw)} for development.")

        question_to_type_map = {}
        if USE_QUESTION_TYPE_FEATURE:
            classifier = get_question_classifier() # Loads the pipeline once
            if classifier:
                logger.info("Pre-computing question types for all unique questions...")
                unique_questions = set()
                for item_list_to_scan in [train_items_raw, dev_items_raw]:
                    for item in item_list_to_scan:
                        unique_questions.add(item['question'])
                
                temp_q_list = list(unique_questions) # More efficient to pass list to pipeline if it supports batching
                # Note: Hugging Face pipeline batching is usually for List[str] not Dataset
                # So we iterate here, but if classifier supports batching, that's better.
                # For now, assuming single prediction per call for get_question_type logic
                for q_idx, q_text in enumerate(temp_q_list):
                    # The get_question_type function handles prediction
                    q_type_pred = get_question_type(q_text) # This uses the global classifier
                    question_to_type_map[q_text] = q_type_pred if q_type_pred else "UNKNOWN_TYPE"

                    if (q_idx + 1) % 200 == 0:
                        logger.info(f"  Predicted types for {q_idx+1}/{len(unique_questions)} unique questions.")
                logger.info(f"Finished pre-computing {len(question_to_type_map)} question types.")
            else:
                logger.warning("Question classifier could not be loaded. USE_QUESTION_TYPE_FEATURE was True, but proceeding without question type features for this run.")
                # Effectively disable it for this processing run if classifier failed
                globals()['USE_QUESTION_TYPE_FEATURE'] = False 
                # This dynamic change is a bit tricky. Better to pass effective_use_q_type to data creation.
                # For now, the create_... functions will check the global var or receive it.

        # Use the global USE_QUESTION_TYPE_FEATURE, which might have been updated if classifier failed
        effective_use_q_type = USE_QUESTION_TYPE_FEATURE and bool(question_classifier_pipeline)

        logger.info(f"Creating training examples (Effective USE_QUESTION_TYPE_FEATURE={effective_use_q_type})...")
        train_examples = create_input_examples_for_cross_encoder( # Keep original name, logic is now conditional inside
            train_items_raw, passages, 
            question_to_type_map=question_to_type_map, # Pass the map
            max_neg_per_positive=5,
            include_q_type=effective_use_q_type # Pass effective status
        )
        if not train_examples:
            raise ValueError("No training examples were created after processing.")

        logger.info(f"Creating development set for Reranking Evaluator (Effective USE_QUESTION_TYPE_FEATURE={effective_use_q_type})...")
        dev_evaluator_data = create_reranking_evaluator_data( # Keep original name
            dev_items_raw, passages, 
            question_to_type_map=question_to_type_map, # Pass the map
            include_q_type=effective_use_q_type # Pass effective status
        )

        # Save the processed data
        os.makedirs(processed_data_path_specific, exist_ok=True)
        train_examples_to_save = [{'texts': ex.texts, 'label': ex.label} for ex in train_examples]
        with open(os.path.join(processed_data_path_specific, "cached_train_examples.json"), 'w', encoding='utf-8') as f:
            json.dump(train_examples_to_save, f, ensure_ascii=False, indent=2)
        with open(os.path.join(processed_data_path_specific, "cached_dev_evaluator_data.json"), 'w', encoding='utf-8') as f:
            json.dump(dev_evaluator_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed training examples and dev evaluator data to {processed_data_path_specific}")

    # ---- Convert train_examples to Hugging Face Dataset ----
    if not train_examples: # Should have been caught earlier if processing failed
        raise ValueError("Training examples list is empty after loading/processing.")
        
    train_texts1 = [ex.texts[0] for ex in train_examples]
    train_texts2 = [ex.texts[1] for ex in train_examples]
    train_labels = [ex.label for ex in train_examples]
    train_hf_dataset = Dataset.from_dict({'sentence1': train_texts1, 'sentence2': train_texts2, 'label': train_labels})
    logger.info(f"Training Hugging Face Dataset created: {train_hf_dataset}")

    # 4. Initialize CrossEncoder model
    # num_labels=1 for regression/ranking score (logit for BCE)
    logger.info(f"Initializing CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}")
    model = CrossEncoder(
        CROSS_ENCODER_MODEL_NAME, 
        num_labels=1, 
        max_length=MAX_SEQ_LENGTH_CROSS_ENCODER,
        trust_remote_code=TRUST_REMOTE_CODE_CROSS_ENCODER,
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Model will be moved by Trainer
    )
    from bigmodelvis import Visualization
    Visualization(model).structure_graph()
    # print(f"Pad token ID sed by collator: {model.tokenizer.pad_token_id}")
    # model.model.config.pad_token_id = model.tokenizer.pad_token_id

    # --- START: Manual Test of CrossEncoder predict method ---
    logger.info("\n--- Starting Manual Test of CrossEncoder.predict() ---")
    if train_hf_dataset and len(train_hf_dataset) > 0:
        # Determine the number of samples for the test batch
        # Use a full batch size if possible, otherwise whatever is available
        num_test_samples_manual = min(TRAIN_BATCH_SIZE, len(train_hf_dataset))

        if num_test_samples_manual == 0:
            logger.warning("Manual test cannot run: train_hf_dataset is empty.")
        else:
            logger.info(f"Taking {num_test_samples_manual} samples from train_hf_dataset for manual test.")

            # Prepare sentence pairs for CrossEncoder.predict()
            # train_hf_dataset has 'sentence1' and 'sentence2' columns
            sentence_pairs_for_manual_test = []
            for i in range(num_test_samples_manual):
                sentence_pairs_for_manual_test.append(
                    (train_hf_dataset[i]['sentence1'], train_hf_dataset[i]['sentence2'])
                )
            
            if sentence_pairs_for_manual_test: # Ensure list is not empty
                logger.info(f"First test pair for manual predict: ('{sentence_pairs_for_manual_test[0][0][:50]}...', '{sentence_pairs_for_manual_test[0][1][:50]}...')")

            # The CrossEncoder's predict method handles tokenization and model forwarding.
            # It's important to test the `model` instance that will be used by the Trainer.
            try:
                # Ensure the model is on the correct device for a realistic test,
                # though .predict() can also handle device placement internally.
                # The Trainer will move the model to args.device.
                # For this test, let's assume the model will use its current device or 'cuda' if available.
                
                logger.info(f"Calling model.predict() with a list of {len(sentence_pairs_for_manual_test)} pairs...")
                with torch.no_grad(): # Ensure inference mode
                    # The `predict` method of CrossEncoder takes a list of sentence pairs.
                    # It has its own internal batching (controlled by its batch_size arg, default 32).
                    # Here, sentence_pairs_for_manual_test is our "dataset" for predict.
                    manual_scores_np = model.predict(
                        sentences=sentence_pairs_for_manual_test,
                        batch_size=TRAIN_BATCH_SIZE, # Tell predict to use this batch size for its internal processing if input > TRAIN_BATCH_SIZE
                        show_progress_bar=False
                    )
                
                manual_scores_tensor = torch.tensor(manual_scores_np)
                
                logger.info(f"MANUAL TEST: Output scores shape: {manual_scores_tensor.shape}")
                logger.info(f"MANUAL TEST: Output scores dtype: {manual_scores_tensor.dtype}")
                logger.info(f"MANUAL TEST: First few scores: {manual_scores_tensor[:min(5, num_test_samples_manual)]}")

                # For num_labels=1, predict() should return a 1D array/tensor of scores
                # with length equal to the number of input sentence pairs.
                if len(manual_scores_tensor.shape) == 1 and manual_scores_tensor.shape[0] == num_test_samples_manual:
                    logger.info(f"SUCCESS (Shape Check): Manual test scores batch dimension ({manual_scores_tensor.shape[0]}) matches number of input samples ({num_test_samples_manual}).")
                elif len(manual_scores_tensor.shape) == 2 and manual_scores_tensor.shape[0] == num_test_samples_manual and manual_scores_tensor.shape[1] == 1:
                    logger.info(f"SUCCESS (Shape Check): Manual test scores shape is [{manual_scores_tensor.shape[0]}, 1], which is also fine and will be squeezed by loss function. Batch dim matches input samples.")
                else:
                    logger.error(f"FAILURE (Shape Check): Manual test scores shape ({manual_scores_tensor.shape}) is UNEXPECTED for {num_test_samples_manual} input samples and num_labels=1.")
                    logger.error("This indicates the problem likely lies in how the CrossEncoder or its underlying model processes batches.")

            except Exception as e:
                logger.error(f"Error during manual CrossEncoder.predict() test: {e}")
                import traceback
                traceback.print_exc()
    else:
        logger.warning("train_hf_dataset is empty or not defined, skipping manual CrossEncoder.predict() test.")
    logger.info("--- Finished Manual Test of CrossEncoder.predict() ---\n")
    # --- END: Manual Test of CrossEncoder Forward Pass ---





    # 5. Define Loss function
    # BinaryCrossEntropyLoss is suitable for (query, passage, 0/1 label) format
    loss_function = losses.BinaryCrossEntropyLoss(model=model)
    logger.info(f"Using loss function: BinaryCrossEntropyLoss")

    # So far 可以正确运行
    # trainer = CrossEncoderTrainer(
    #     model=model,
    #     train_dataset=train_hf_dataset,
    #     loss=loss_function,
    # )
    # trainer.train()

    # 6. Prepare data for CrossEncoderRerankingEvaluator
    if not dev_evaluator_data:
        logger.warning("Development set for evaluator is empty. Evaluation might not run.")
        # Create a dummy evaluator or handle this case if dev_items could be empty
        dev_evaluator = None
    else:
        import random
        print(dev_evaluator_data[0]) # Debug print to check the first item
        length = len(dev_evaluator_data)
        for i, item in enumerate(dev_evaluator_data):
            # 如果没有 negative，就从其他句子中随机抽取一些positive作为 negative
            negatives = item.get('negative', [])
            if len(negatives) == 0:
                # 从所有正面句子中随机抽取一些作为负面句子
                
                dev_evaluator_data[i]['negative'] = [
                    dev_evaluator_data[i]['positive'][0]
                    for i in random.sample(range(length), 3)
                ]

        dev_evaluator = CrossEncoderRerankingEvaluator(
            samples=dev_evaluator_data,
            name='dev_reranker',
            # mrr_at_k=[1, 3, 5, 10], # Configure which MRR@k values to compute
            at_k = 10,
            write_csv=True,
            batch_size=EVAL_BATCH_SIZE,
            show_progress_bar=True,
        )

    # 7. Define Training Arguments
    # Training arguments
    # The model name for run_name should not contain '/'
    safe_model_name_for_run = CROSS_ENCODER_MODEL_NAME.split('/')[-1]
    run_name = f"ranker-{safe_model_name_for_run}-qtype_{USE_QUESTION_TYPE_FEATURE}"

    args = CrossEncoderTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR_RANKER, "training_output", run_name),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_STEPS_RATIO,
        fp16=torch.cuda.is_available(),
        # bf16=False, # Set to True if your GPU supports BF16
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save model at the end of each epoch
        save_total_limit=2,
        load_best_model_at_end=True, # Requires eval_strategy and metric_for_best_model
        metric_for_best_model="dev_reranker_mrr@10", # Example: Use MRR@10 from dev_reranker
        logging_steps=max(1, len(train_hf_dataset) // (TRAIN_BATCH_SIZE * 10) if TRAIN_BATCH_SIZE > 0 else 100), # Log ~10 times per epoch
        run_name=run_name,
        # dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 2 # Optional: for data loading
    )
    # print(train_hf_dataset)
    # print(model)
    # print(loss_function)
    print(train_hf_dataset[0])

    # 8. Create Trainer and Train
    logger.info("Initializing CrossEncoderTrainer...")

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_hf_dataset,
        eval_dataset=None, # We use evaluator for dev set metrics
        loss=loss_function,
        evaluator=dev_evaluator, # Pass the evaluator here
        # tokenizer=model.tokenizer # Trainer can usually infer this from model
    )
    
    logger.info(f"Starting CrossEncoder training for {NUM_EPOCHS} epochs...")
    trainer.train()

    # 9. Save the final model
    final_save_path = os.path.join(OUTPUT_DIR_RANKER, "final_model")
    logger.info(f"Training complete. Saving final model to {final_save_path}")
    model.save_pretrained(final_save_path)
    # The tokenizer is part of the CrossEncoder and saved with it.

    # 10. (Optional) Final evaluation on dev set with the best loaded model
    if dev_evaluator:
        logger.info("Running final evaluation with the best model on the dev set...")
        # The trainer.model should be the best model if load_best_model_at_end=True
        final_metrics = dev_evaluator(trainer.model, output_path=OUTPUT_DIR_RANKER) 
        logger.info(f"Final dev set metrics: {final_metrics}")
        # MRR value might be in a nested dict or specific key e.g. dev_reranker_MRR@10
        # The CrossEncoderRerankingEvaluator saves results to a CSV by default if write_csv=True
        # and returns a dictionary of scores.

    logger.info(f"Script finished. Check outputs in {OUTPUT_DIR_RANKER}")