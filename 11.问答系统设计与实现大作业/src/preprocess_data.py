import jieba
import json

# Define file paths (assuming the script is run from the project root)
STOPWORDS_PATH = 'data/stopwords.txt'
PASSAGES_PATH = 'data/passages_multi_sentences.json'
TRAIN_DATA_PATH = 'data/train.json'

def load_stopwords(filepath):
    """Loads stopwords from a file into a set."""
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    print(f"Loaded {len(stopwords)} stopwords.")
    return stopwords

def preprocess_text(text, stopwords_set):
    """
    Segments text using Jieba, removes stopwords, and filters punctuation.
    Args:
        text (str): The input text string.
        stopwords_set (set): A set of stopwords.
    Returns:
        list: A list of processed words.
    """
    seg_list = jieba.lcut(text)
    processed_words = []
    for word in seg_list:
        word = word.strip() # Remove leading/trailing whitespace
        if word and word not in stopwords_set and not word.isspace(): # Add more conditions if needed (e.g. word.isalnum() to keep only alphanumeric)
            # Here you might want to add more checks, e.g., for purely punctuation words
            # or to convert to lowercase if desired.
            # A simple check for non-empty and not purely space after stripping.
            processed_words.append(word)
    return processed_words

if __name__ == '__main__':
    # Load stopwords
    stopwords = load_stopwords(STOPWORDS_PATH)

    # --- Process Passages ---
    processed_passages = {} # To store pid -> list of processed sentences (each a list of words)
    print(f"\nProcessing passages from {PASSAGES_PATH}...")
    with open(PASSAGES_PATH, 'r', encoding='utf-8') as f_passages:
        for line_num, line in enumerate(f_passages):
            try:
                record = json.loads(line.strip())
                pid = record.get('pid')
                document_sentences = record.get('document', []) # This is a list of sentences [cite: 1]

                if pid is None:
                    print(f"Warning: Missing 'pid' in line {line_num+1} of {PASSAGES_PATH}")
                    continue

                processed_doc = []
                for sentence in document_sentences:
                    processed_sentence_words = preprocess_text(sentence, stopwords)
                    if processed_sentence_words: # Only add if not empty
                        processed_doc.append(processed_sentence_words)
                
                processed_passages[pid] = processed_doc
                
                if (line_num + 1) % 1000 == 0: # Log progress
                    print(f"Processed {line_num + 1} passages...")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {PASSAGES_PATH} at line {line_num+1}: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred processing line {line_num+1} in {PASSAGES_PATH}: {e}")


    print(f"Finished processing {len(processed_passages)} passages.")
    # You might want to save processed_passages to a file or use it directly.
    # For example, to see the first processed passage:
    if processed_passages:
        first_pid = list(processed_passages.keys())[0]
        print(f"\nExample processed passage (pid: {first_pid}):")
        for i, sent_words in enumerate(processed_passages[first_pid][:2]): # Print first 2 sentences
             print(f"  Sentence {i+1}: {sent_words}")


    # --- Process Questions from train.json ---
    processed_questions = {} # To store qid -> list of processed question words
    print(f"\nProcessing questions from {TRAIN_DATA_PATH}...")
    question_count = 0
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f_train:
        for line_num, line in enumerate(f_train):
            try:
                record = json.loads(line.strip())
                qid = record.get('qid')
                question_text = record.get('question')

                if qid is None or question_text is None:
                    print(f"Warning: Missing 'qid' or 'question' in line {line_num+1} of {TRAIN_DATA_PATH}")
                    continue
                
                processed_question_words = preprocess_text(question_text, stopwords)
                processed_questions[qid] = processed_question_words
                question_count +=1

                if (line_num + 1) % 1000 == 0: # Log progress
                    print(f"Processed {line_num + 1} questions...")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {TRAIN_DATA_PATH} at line {line_num+1}: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred processing line {line_num+1} in {TRAIN_DATA_PATH}: {e}")
                
    print(f"Finished processing {question_count} questions.")
    # Example of a processed question:
    if processed_questions:
        first_qid = list(processed_questions.keys())[0]
        print(f"\nExample processed question (qid: {first_qid}): {processed_questions[first_qid]}")

