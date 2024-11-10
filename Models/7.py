#0.49
import json
import logging
import time
from pathlib import Path
from scapy.all import rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import joblib
import os
from sklearn.naive_bayes import BernoulliNB

# Set up logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')

def extract_events_from_pcap(file_path):
    """Extract events from PCAP files with optimized JSON decoding."""
    try:
        packets = rdpcap(file_path)
        events = []

        for pkt in packets:
            if pkt.haslayer('TCP') and pkt['TCP'].payload:
                try:
                    payload = pkt['TCP'].payload.load.decode('utf-8')
                    event = json.loads(payload)
                    if isinstance(event, dict):
                        events.append(event)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        return events
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return []

def flatten_event(event):
    """Flatten nested structures in events."""
    flattened = {}
    for key, value in event.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened[f"{key}_{sub_key}"] = sub_value
        else:
            flattened[key] = value
    return flattened

def process_files_in_batches(input_dir, batch_size=20):
    """Process files in larger batches to optimize memory usage."""
    files = list(input_dir.iterdir())
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        data, labels = [], []
        with ProcessPoolExecutor() as executor:
            results = executor.map(extract_events_from_pcap, (str(file) for file in batch_files))
        for events in results:
            for event in events:
                event = flatten_event(event)
                data.append(event)
                labels.append(event.get('label', 0))
        yield data, labels

def vectorize_data(data_batches, tfidf_vectorizer, count_vectorizer):
    """Vectorize the data using both TF-IDF and CountVectorizer."""
    try:
        tfidf_X = tfidf_vectorizer.transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])
        count_X = count_vectorizer.transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])
        return tfidf_X, count_X
    except Exception as e:
        logging.error(f"Error during vectorization: {e}")
        return None, None

def main():
    # Process and flatten data in batches
    data_batches, labels_batches = [], []
    for data_batch, labels_batch in process_files_in_batches(INPUT_DIR / "train", batch_size=50):
        data_batches.extend(data_batch)
        labels_batches.extend(labels_batch)

    # Use only TF-IDF with limited features for simplicity
    tfidf_vectorizer = TfidfVectorizer(max_features=500, max_df=0.85, min_df=5, stop_words='english')
    tfidf_vectorizer.fit([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])

    tfidf_X_train = tfidf_vectorizer.transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])

    # Initialize a very simple BernoulliNB model
    model = BernoulliNB()

    start_time = time.time()
    model.fit(tfidf_X_train, labels_batches)

    test_data_batches = []
    for data_batch, _ in process_files_in_batches(INPUT_DIR / 'test', batch_size=50):
        test_data_batches.extend(data_batch)

    tfidf_X_test = tfidf_vectorizer.transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in test_data_batches])

    # Predict probabilities and adjust threshold to favor '1's
    test_probs = model.predict_proba(tfidf_X_test)[:, 1]
    predictions = (test_probs > 0.4).astype(int)  # Lowering threshold to 0.4 to bias towards '1's

    # Save results as labels
    labels = {file.name: int(pred) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

    # Save the model and vectorizer
    logging.info("Saving model, vectorizer, and metadata...")
    joblib.dump(model, OUTPUT_DIR / 'bernoulli_nb_model.pkl')
    joblib.dump(tfidf_vectorizer, OUTPUT_DIR / 'tfidf_vectorizer.pkl')

    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'threshold': 0.4,
            'tfidf_feature_names': tfidf_vectorizer.get_feature_names_out()
        }, f)

if __name__ == "__main__":
    main()
