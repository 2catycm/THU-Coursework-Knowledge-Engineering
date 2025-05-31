import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Dict, List, Tuple
import torch
import shutil # For removing cache directory if needed

# --- Configuration ---
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Model for document embeddings (SBERT)
# Using a common Chinese sentence transformer.
# DMetaSoul/sbert-chinese-general-v1 is RoFormer based, ensure SentenceTransformer can load it.
# shibing624/text2vec-base-chinese is another good option directly compatible with SentenceTransformer.
# SBERT_MODEL_NAME = "shibing624/text2vec-base-chinese"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# SBERT_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v1" # Alternative, if it loads well with SentenceTransformer

# Data paths
PASSAGES_FILE_PATH = "./data/passages_multi_sentences.json"
OUTPUT_SEMANTIC_IDS_FILE = "./data/semantic_identifiers.json"
EMBEDDINGS_CACHE_PATH = "./data/document_embeddings_cache.npz" # Cache for embeddings

# Hierarchical K-Means parameters
# 'c' from the problem description: number of clusters at each k-means step,
# and also the threshold for deciding to recurse (if a cluster has more than 'c' items).
# Path components will be 0 to c-1.
C_PARAM = 10  # Number of clusters (k) for each KMeans step, and also the split threshold.
              # A cluster with > C_PARAM documents will be split further.
MAX_RECURSION_DEPTH = 5   # Max depth of the hierarchy to prevent overly long IDs
MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS = 2 # Minimum samples needed to run KMeans (must be >= k)

# Ensure C_PARAM is reasonable for K-Means
if MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS < C_PARAM:
    logger.warning(f"MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS ({MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS}) is less than C_PARAM ({C_PARAM}). "
                   f"Adjusting MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS to be C_PARAM for K-Means to run.")
    MIN_SAMPLES_IN_CLUSTER_FOR_KMEANS = C_PARAM


FORCE_RECOMPUTE_EMBEDDINGS = False # Set to True to recompute embeddings even if cache exists

# --- Helper Functions ---

def load_passages_for_clustering(filepath: str) -> Dict[str, str]:
    """
    Loads passages. Value is the concatenation of all sentences in a document.
    """
    logger.info(f"Loading passages from {filepath}...")
    passages_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                pid = str(data['pid'])
                # Concatenate sentences to form a single document text
                doc_text = " ".join(data.get('document', [])).strip()
                if doc_text: # Only include if document text is not empty
                    passages_dict[pid] = doc_text
        logger.info(f"Loaded {len(passages_dict)} non-empty documents.")
    except FileNotFoundError:
        logger.error(f"Passages file not found at {filepath}.")
        raise
    return passages_dict

def get_document_embeddings(
    doc_texts_dict: Dict[str, str], 
    model_name: str, 
    cache_path: str, 
    force_recompute: bool = False
) -> Tuple[List[str], np.ndarray]:
    """
    Generates or loads document embeddings using SBERT.
    Returns:
        pids_list (List[str]): Ordered list of PIDs.
        embeddings_matrix (np.ndarray): Matrix of embeddings, order matches pids_list.
    """
    if not force_recompute and os.path.exists(cache_path):
        logger.info(f"Loading embeddings from cache: {cache_path}")
        try:
            cached_data = np.load(cache_path, allow_pickle=True)
            pids_list = cached_data['pids_list'].tolist()
            embeddings_matrix = cached_data['embeddings_matrix']
            # Sanity check
            if len(pids_list) == embeddings_matrix.shape[0] and \
               all(pid in doc_texts_dict for pid in pids_list) and \
               len(pids_list) == len(doc_texts_dict): # Ensure cached pids match current docs
                logger.info(f"Successfully loaded {len(pids_list)} embeddings from cache.")
                return pids_list, embeddings_matrix
            else:
                logger.warning("Cache mismatch or inconsistency. Recomputing embeddings.")
        except Exception as e:
            logger.warning(f"Could not load embeddings from cache due to {e}. Recomputing.")

    logger.info(f"Generating document embeddings using {model_name}...")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model '{model_name}': {e}")
        logger.error("Ensure the model name is correct and sentence-transformers is installed.")
        raise

    pids_list = list(doc_texts_dict.keys())
    doc_texts_list = [doc_texts_dict[pid] for pid in pids_list]
    
    # Generate embeddings
    # Adjust batch_size based on your GPU memory
    embeddings_matrix = model.encode(doc_texts_list, show_progress_bar=True, batch_size=32)
    
    logger.info(f"Generated {len(embeddings_matrix)} embeddings with dimension {embeddings_matrix.shape[1]}.")
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, pids_list=np.array(pids_list, dtype=object), embeddings_matrix=embeddings_matrix)
        logger.info(f"Embeddings saved to cache: {cache_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings to cache: {e}")
        
    return pids_list, np.array(embeddings_matrix)


# --- Hierarchical K-Means ---
# Global dictionary to store the final semantic ID for each PID
# pid_to_semantic_id_map: Dict[str, List[int]] = {} # Initialized in main

def assign_leaf_node_ids(
    pids_in_leaf_node: List[str],
    current_path: List[int],
    semantic_id_map_accumulator: Dict[str, List[int]]
):
    """Assigns the current_path as the semantic ID for all pids in a leaf node."""
    for pid in pids_in_leaf_node:
        semantic_id_map_accumulator[pid] = list(current_path) # Store a copy

def hierarchical_cluster_and_assign_ids(
    pids_for_current_node: List[str],
    embeddings_for_current_node: np.ndarray,
    k_per_level: int, # 'c' from problem: number of clusters, path components are 0 to k-1
    split_decision_threshold: int, # 'c' from problem: if len(cluster) > this, split
    max_depth: int,
    current_path: List[int],
    current_depth: int,
    semantic_id_map_accumulator: Dict[str, List[int]]
):
    """
    Recursively performs K-Means and assigns semantic IDs.
    - k_per_level: Number of clusters for KMeans at this step.
    - split_decision_threshold: If a formed cluster has more items than this, it's split further.
    """
    node_doc_count = len(pids_for_current_node)
    logger.debug(f"Depth {current_depth}, Path {'-'.join(map(str,current_path)) if current_path else 'ROOT'}, Docs: {node_doc_count}")

    # Termination conditions for recursion
    if current_depth >= max_depth:
        logger.debug(f"  Max depth {max_depth} reached. Assigning path {'-'.join(map(str,current_path))} to {node_doc_count} docs.")
        assign_leaf_node_ids(pids_for_current_node, current_path, semantic_id_map_accumulator)
        return
    
    if node_doc_count <= split_decision_threshold:
        # This node is small enough, or it's a leaf because it can't be meaningfully split by k-means.
        # The problem implies "如果一簇文档数量超过c，就递归". So if <=c, it's a leaf for path purposes.
        logger.debug(f"  Node size {node_doc_count} <= split_decision_threshold {split_decision_threshold}. Assigning path {'-'.join(map(str,current_path))} to {node_doc_count} docs.")
        assign_leaf_node_ids(pids_for_current_node, current_path, semantic_id_map_accumulator)
        return

    # Ensure enough samples for K-Means (sklearn needs n_samples >= n_clusters)
    if node_doc_count < k_per_level:
        logger.debug(f"  Node size {node_doc_count} < k_per_level {k_per_level}. Cannot perform K-Means. Assigning path {'-'.join(map(str,current_path))} to {node_doc_count} docs.")
        assign_leaf_node_ids(pids_for_current_node, current_path, semantic_id_map_accumulator)
        return

    # Perform K-Means clustering for the current node
    logger.debug(f"  Performing K-Means (k={k_per_level}) on {node_doc_count} docs.")
    try:
        kmeans = KMeans(n_clusters=k_per_level, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings_for_current_node)
    except Exception as e:
        logger.error(f"Error during K-Means at path {'-'.join(map(str,current_path))}: {e}. Treating as leaf.")
        assign_leaf_node_ids(pids_for_current_node, current_path, semantic_id_map_accumulator)
        return

    # Process each new sub-cluster
    for i in range(k_per_level):
        sub_cluster_indices = np.where(cluster_labels == i)[0]
        
        if len(sub_cluster_indices) == 0: # Empty cluster
            continue

        pids_in_sub_cluster = [pids_for_current_node[idx] for idx in sub_cluster_indices]
        embeddings_for_sub_cluster = embeddings_for_current_node[sub_cluster_indices]
        new_path = current_path + [i] # Append the current sub-cluster label to the path
        
        # Recursive call for the sub-cluster
        hierarchical_cluster_and_assign_ids(
            pids_in_sub_cluster,
            embeddings_for_sub_cluster,
            k_per_level,
            split_decision_threshold,
            max_depth,
            new_path,
            current_depth + 1,
            semantic_id_map_accumulator
        )

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Task 1: Generating Semantic Identifiers for Documents ---")

    # 1. Load passages
    passages_content_dict = load_passages_for_clustering(PASSAGES_FILE_PATH)
    if not passages_content_dict:
        logger.error("No passages loaded. Exiting.")
        exit()

    # 2. Get document embeddings
    # This returns pids in a fixed order and the corresponding embedding matrix
    ordered_pids, all_embeddings_matrix = get_document_embeddings(
        passages_content_dict, SBERT_MODEL_NAME, EMBEDDINGS_CACHE_PATH, FORCE_RECOMPUTE_EMBEDDINGS
    )
    if len(ordered_pids) == 0:
        logger.error("No embeddings generated. Exiting.")
        exit()

    # 3. Perform hierarchical K-Means and assign semantic IDs
    logger.info("Starting hierarchical K-Means clustering...")
    pid_to_semantic_id_map: Dict[str, List[int]] = {} # Initialize the global map

    # The problem statement "对每一簇数量为c的文档集中的每个文档分配一个编号（0，c-1）" and
    # "如果一簇文档数量超过c，就递归地进行k-means聚类" implies that 'c' is used for both
    # k in KMeans (n_clusters=c) and the threshold for recursion (if size > c).
    # So, k_per_level = C_PARAM and split_decision_threshold = C_PARAM.
    
    hierarchical_cluster_and_assign_ids(
        pids_for_current_node=ordered_pids,
        embeddings_for_current_node=all_embeddings_matrix,
        k_per_level=C_PARAM,
        split_decision_threshold=C_PARAM, # Recurse if cluster_size > C_PARAM
        max_depth=MAX_RECURSION_DEPTH,
        current_path=[],
        current_depth=0,
        semantic_id_map_accumulator=pid_to_semantic_id_map
    )

    logger.info(f"Generated semantic IDs for {len(pid_to_semantic_id_map)} documents.")

    # 4. Save the semantic IDs
    # Convert path lists to strings like "r0-r1-r2" for easier use in T5
    semantic_ids_str_map = {pid: "-".join(map(str, path)) for pid, path in pid_to_semantic_id_map.items()}
    
    try:
        os.makedirs(os.path.dirname(OUTPUT_SEMANTIC_IDS_FILE), exist_ok=True)
        with open(OUTPUT_SEMANTIC_IDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(semantic_ids_str_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Semantic identifiers saved to: {OUTPUT_SEMANTIC_IDS_FILE}")
    except Exception as e:
        logger.error(f"Error saving semantic identifiers: {e}")

    # Optional: Print some examples
    logger.info("\n--- Example Semantic Identifiers ---")
    count = 0
    for pid, sem_id_str in semantic_ids_str_map.items():
        logger.info(f"PID: {pid}, Semantic ID: {sem_id_str}")
        count += 1
        if count >= 5:
            break
            
    logger.info("--- Task 1 Finished ---")