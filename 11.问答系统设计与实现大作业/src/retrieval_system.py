import os
import json
import jieba
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import Tokenizer, Token
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.searching import Searcher

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
    def __call__(self, text, **kargs):
        words = jieba.lcut(text) # Use lcut for precise mode, returns list
        token = Token()
        for word in words:
            word_clean = word.strip()
            if word_clean and word_clean.lower() not in STOPWORDS and not word_clean.isspace():
                token.original = token.text = word_clean
                token.boost = 1.0
                token.pos = None # Not tracking position for simplicity here
                token.removed = False
                yield token

def chinese_analyzer():
    """Returns a custom Whoosh Analyzer for Chinese text."""
    return ChineseTokenizer()

def preprocess_question(question_text):
    """
    Preprocesses a question string: segmentation, stop-word removal, deduplication.
    (Addresses part of step a for questions and part of step c for keyword extraction)
    """
    seg_list = jieba.lcut(question_text)
    keywords = []
    seen_keywords = set()
    for word in seg_list:
        word_clean = word.strip().lower() # Lowercase for consistency
        if word_clean and word_clean not in STOPWORDS and not word_clean.isspace():
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

# --- Step c) Querying the Index ---
def search_documents(ix, question_text, top_n=3):
    """
    Searches the index for a given question and returns top_n pids.
    (Addresses part of step c)
    """
    # Preprocess the question to get keywords (also part of step c)
    query_keywords = preprocess_question(question_text)
    if not query_keywords:
        return []

    # Form a query string (Whoosh OR query by default for space-separated terms)
    # Using OR logic, as suggested by "命中一个关键词加一" then sum scores.
    # Whoosh's default OR for space-separated terms is a good starting point.
    query_str = " ".join(query_keywords)
    
    retrieved_pids = []
    with ix.searcher() as searcher:
        # Using MultifieldParser to search in the 'content' field.
        # You could use QueryParser for a single field too.
        # Default operator is OR. To make it AND, use qparser.AND
        query_parser = QueryParser("content", schema=ix.schema)
        query = query_parser.parse(query_str)
        
        results = searcher.search(query, limit=top_n)
        for hit in results:
            retrieved_pids.append(hit['pid']) # pid is stored as string in index
            
    return retrieved_pids

# --- Step d) Evaluating Retrieval ---
def evaluate_retrieval(ix, train_filepath):
    """
    Evaluates the retrieval system using questions from train_filepath.
    Calculates Top1 and Top3 accuracy.
    (Addresses step d)
    """
    print(f"\nEvaluating retrieval performance using {train_filepath}...")
    total_questions = 0
    top1_correct = 0
    top3_correct = 0

    with open(train_filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                question_text = record.get('question')
                # Assuming 'pid' in train.json is the single ground truth document ID
                # The problem description implies 'pid' in train.json is the correct one
                # "利用train.json中正确的pid以及检索系统返回的前三个pid来计算检索的准确率"
                ground_truth_pid_list = record.get('pid') 
                
                # Ensure ground_truth_pid is a list of strings for consistent comparison
                if ground_truth_pid_list is None or question_text is None:
                    print(f"Warning: Missing 'question' or 'pid' in line {line_num+1} of {train_filepath}")
                    continue

                # The problem states pid is an int in train.json, convert to string for comparison
                # and handle if it's a list (though seems like a single int usually)
                if isinstance(ground_truth_pid_list, int):
                    ground_truth_pids = {str(ground_truth_pid_list)}
                elif isinstance(ground_truth_pid_list, list): # If pid can be a list of relevant pids
                    ground_truth_pids = {str(p) for p in ground_truth_pid_list}
                else: # Fallback if it's already a string
                     ground_truth_pids = {str(ground_truth_pid_list)}


                if not ground_truth_pids:
                    print(f"Warning: No ground truth PID for question at line {line_num+1}")
                    continue

                total_questions += 1
                
                # Perform search (Step c is implicitly called here)
                retrieved_pids = search_documents(ix, question_text, top_n=3)

                if not retrieved_pids:
                    if (line_num + 1) % 200 == 0: # Log progress less frequently during eval
                        print(f"  Evaluated {total_questions} questions...")
                    continue

                # Check Top1 accuracy
                if retrieved_pids[0] in ground_truth_pids:
                    top1_correct += 1
                
                # Check Top3 accuracy
                found_in_top3 = False
                for pid in retrieved_pids: # retrieved_pids already has at most 3 elements
                    if pid in ground_truth_pids:
                        found_in_top3 = True
                        break
                if found_in_top3:
                    top3_correct += 1
                
                if total_questions % 200 == 0: # Log progress less frequently during eval
                    print(f"  Evaluated {total_questions} questions...")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {train_filepath} at line {line_num+1}")
            except Exception as e:
                print(f"An error occurred evaluating line {line_num+1}: {e}")


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
    # Step a & b: Build or load the index
    # Set force_rebuild=True if you want to recreate the index from scratch
    index = build_index(PASSAGES_FILE, INDEX_DIR, force_rebuild=False)

    # Example of how to use search_documents (Step c)
    if index:
        sample_question = "清朝的开国皇帝是谁？" # Replace with an actual question if desired
        print(f"\nSearching for a sample question: '{sample_question}'")
        sample_retrieved_docs = search_documents(index, sample_question, top_n=3)
        print(f"Retrieved PIDs for sample question: {sample_retrieved_docs}")

        # Step d: Evaluate the retrieval system
        evaluate_retrieval(index, TRAIN_FILE)
    else:
        print("Failed to build or load the index. Evaluation cannot proceed.")