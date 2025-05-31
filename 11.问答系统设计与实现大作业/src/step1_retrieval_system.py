import json
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import Tokenizer, Token
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.searching import Searcher

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


# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in src
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
INDEX_DIR = os.path.join(os.path.dirname(BASE_DIR), 'indexdir') # Place indexdir alongside data and src

PASSAGES_FILE = os.path.join(DATA_DIR, 'passages_multi_sentences.json')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
STOPWORDS_FILE = os.path.join(DATA_DIR, 'stopwords.txt')

# --- Utility Functions ---

def load_stopwords(filepath):
    """Loads stopwords from a file into a set."""
    stopwords_set = set()
    if not os.path.exists(filepath):
        print(f"Warning: Stopwords file not found at {filepath}. Proceeding without stopwords.")
        return stopwords_set
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords_set.add(line.strip())
    print(f"Loaded {len(stopwords_set)} stopwords.")
    return stopwords_set

# Global stopwords set
STOPWORDS = load_stopwords(STOPWORDS_FILE)

# --- Step a) Part 1: Preprocessing logic for Whoosh Analyzer and Questions ---
class ChineseTokenizer(Tokenizer):
    """Custom Whoosh Tokenizer for Chinese text using Jieba."""
    def __call__(self, text:str, **kargs):
        words = jieba.lcut(text) 
        token_instance = Token() # Create one Token object to reuse
        current_pos = 0
        for word in words:
            word_clean = word.strip()
            # Ensure it's not a stopword and not just whitespace
            if word_clean and word_clean.lower() not in STOPWORDS and not word_clean.isspace():
                token_instance.text = word_clean
                token_instance.original = word_clean
                token_instance.pos = current_pos # Assign current position
                # token_instance.startchar = start_char # Optional: if you track character offsets
                # token_instance.endchar = end_char   # Optional
                yield token_instance
                current_pos += 1

def chinese_analyzer():
    """Returns a custom Whoosh Analyzer for Chinese text."""
    return ChineseTokenizer()

import jieba.posseg as pseg # Import pseg for POS tagging


def preprocess_question(question_text, pos_tags_to_keep=None):
    """
    Preprocesses a question string using POS tagging to extract keywords.
    Args:
        question_text (str): The input question.
        pos_tags_to_keep (set, optional): A set of POS tags to consider for keywords. 
                                         Defaults to nouns, proper nouns, and verbs.
    Returns:
        list: A list of deduplicated keywords.
    """
    if pos_tags_to_keep is None:
        pos_tags_to_keep = {'n', 'nr', 'ns', 'nt', 'nz', 'v'} # Nouns, proper nouns, verbs

    words_with_pos = pseg.lcut(question_text) # List of (word, flag) pairs
    keywords = []
    seen_keywords = set()

    for word, flag in words_with_pos:
        word_clean = word.strip().lower()
        # Check if the word is not a stopword, not just whitespace,
        # and its POS tag (or the beginning of it, e.g., 'nrfg' starts with 'nr') is in our desired set.
        if word_clean and word_clean not in STOPWORDS and not word_clean.isspace():
            is_kept_pos = False
            for kept_tag in pos_tags_to_keep:
                if flag.startswith(kept_tag):
                    is_kept_pos = True
                    break
            
            if is_kept_pos:
                if word_clean not in seen_keywords:
                    keywords.append(word_clean)
                    seen_keywords.add(word_clean)
    return keywords

# --- Step b) Indexing Documents ---
def build_index(passages_filepath, index_dir_path, force_rebuild=False):
    """Builds or opens a Whoosh index for the documents."""
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)

    # Define the schema
    schema = Schema(
        pid=ID(stored=True, unique=True),
        content=TEXT(analyzer=chinese_analyzer(), stored=False) # Content not stored to save space
    )

    if exists_in(index_dir_path) and not force_rebuild:
        print(f"Opening existing index in {index_dir_path}...")
        ix = open_dir(index_dir_path)
    else:
        print(f"Creating new index in {index_dir_path}...")
        ix = create_in(index_dir_path, schema)
        writer = ix.writer(procs=os.cpu_count(), multivalue=True, limitmb=512) # Adjusted writer params

        print(f"Indexing documents from {passages_filepath}...")
        doc_count = 0
        with open(passages_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    pid = record.get('pid')
                    document_sentences = record.get('document', []) # List of sentences

                    if pid is None:
                        print(f"Warning: Missing 'pid' in line {line_num+1} of {passages_filepath}")
                        continue

                    # Combine sentences into a single string for indexing
                    # This completes the document preprocessing part of step a)
                    full_document_text = " ".join(document_sentences)

                    writer.add_document(pid=str(pid), content=full_document_text)
                    doc_count += 1
                    if doc_count % 1000 == 0:
                        print(f"  Indexed {doc_count} documents...")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {passages_filepath} at line {line_num+1}")
                except Exception as e:
                    print(f"An error occurred processing line {line_num+1}: {e}")
        
        print(f"Committing {doc_count} documents to index...")
        writer.commit()
        print("Index building complete.")
    return ix

import whoosh
# --- Step c) Querying the Index ---
def search_documents(searcher, query_parser, question_text, top_n=3, pos_tags_to_keep=None): # Added searcher and query_parser
    """
    Searches the index for a given question using provided searcher and query_parser.
    """
    # Preprocess the question to get keywords
    query_keywords = preprocess_question(question_text, pos_tags_to_keep=pos_tags_to_keep) # POS tagging still happens here
    if not query_keywords:
        return []

    # debug
    # print(query_keywords)
    query_str = " ".join(query_keywords)
    
    retrieved_pids = []
    try:
        query = query_parser.parse(query_str)
        results = searcher.search(query, limit=top_n)
        for hit in results:
            retrieved_pids.append(hit['pid'])
    except Exception as e: # Catch parsing errors or other search issues
        # print(f"Error searching for query '{query_str}': {e}") # Optional: log search errors
        pass # Return empty list if search fails
            
    return retrieved_pids

# --- Step d) Evaluating Retrieval ---
def evaluate_retrieval(ix, train_filepath, pos_tags_to_keep=None): # Pass pos_tags_to_keep if you want to configure it
    """
    Evaluates the retrieval system using questions from train_filepath.
    Calculates Top1 and Top3 accuracy.
    """
    print(f"\nEvaluating retrieval performance using {train_filepath}...")
    total_questions = 0
    top1_correct = 0
    top3_correct = 0

    # Create Searcher and QueryParser once before the loop
    with ix.searcher() as searcher: # Open searcher once
        query_parser = QueryParser("content", schema=ix.schema, group=whoosh.qparser.OrGroup) # Or AndGroup if you prefer

        with open(train_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    question_text = record.get('question')
                    ground_truth_pid_list = record.get('pid') 
                    
                    if ground_truth_pid_list is None or question_text is None:
                        # ... (error handling as before) ...
                        continue

                    if isinstance(ground_truth_pid_list, int):
                        ground_truth_pids = {str(ground_truth_pid_list)}
                    elif isinstance(ground_truth_pid_list, list): 
                        ground_truth_pids = {str(p) for p in ground_truth_pid_list}
                    else: 
                        ground_truth_pids = {str(ground_truth_pid_list)}

                    if not ground_truth_pids:
                        # ... (error handling as before) ...
                        continue
                    
                    total_questions += 1
                    
                    # Call search_documents with the existing searcher and query_parser
                    retrieved_pids = search_documents(searcher, query_parser, question_text, top_n=3, pos_tags_to_keep=pos_tags_to_keep)

                    # ... (rest of the evaluation logic: Top1, Top3 checks) ...
                    if not retrieved_pids:
                        if (total_questions) % 200 == 0: 
                            print(f"  Evaluated {total_questions} questions...")
                        continue

                    if retrieved_pids[0] in ground_truth_pids:
                        top1_correct += 1
                    
                    found_in_top3 = False
                    for pid_ret in retrieved_pids: 
                        if pid_ret in ground_truth_pids:
                            found_in_top3 = True
                            break
                    if found_in_top3:
                        top3_correct += 1
                    
                    if total_questions % 200 == 0: 
                        print(f"  Evaluated {total_questions} questions...")

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {train_filepath} at line {line_num+1}")
                except Exception as e:
                    print(f"An error occurred evaluating line {line_num+1}: {e}")

    # ... (print results as before) ...
    if total_questions == 0:
        print("No questions were evaluated.")
        return 0.0, 0.0

    top1_accuracy = top1_correct / total_questions
    top3_accuracy = top3_correct / total_questions

    print("\n--- Retrieval Evaluation Results ---")
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_correct}/{total_questions})")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_correct}/{total_questions})")
    
    return top1_accuracy, top3_accuracy

# --- Main Execution ---
if __name__ == '__main__':
    index = build_index(PASSAGES_FILE, INDEX_DIR, force_rebuild=False)

    if index:
        # Define your desired POS tags for keyword extraction globally or pass them
        default_pos_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'v'}
        
        # Example search (will need a temporary searcher or adapt search_documents further if used standalone)
        # For standalone search, you might revert search_documents or create a new one:
        sample_question = "2014年南京青年奥林匹克运动会有哪些项目？"
        # sample_question = "南京 奥林匹克运动会 哪些项目"
        print(f"\nSearching for a sample question: '{sample_question}'")
        print("关键词：", preprocess_question(sample_question, pos_tags_to_keep=default_pos_tags))
        with index.searcher() as s:
           qp = QueryParser("content", schema=index.schema, group=whoosh.qparser.OrGroup)
           sample_retrieved_docs = search_documents(s, qp, sample_question, top_n=3, pos_tags_to_keep=default_pos_tags)
        #    sample_retrieved_docs = search_documents(s, qp, "奥林匹克运动会", top_n=3, pos_tags_to_keep=default_pos_tags)

        print(f"Retrieved PIDs for sample question: {sample_retrieved_docs}")

        evaluate_retrieval(index, TRAIN_FILE) # Pass pos_tags_to_keep if you want to customize from main
    else:
        print("Failed to build or load the index. Evaluation cannot proceed.")