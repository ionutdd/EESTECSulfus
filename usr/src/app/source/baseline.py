import json
import logging
import time
from pathlib import Path
from scapy.all import rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import joblib
import os

# Set up logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')

def extract_events_from_pcap(file_path):
    """Extract events from PCAP files with optimized JSON decoding."""
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

def flatten_event(event):
    """Flatten nested structures in events with optimized handling for nested keys."""
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
        with ThreadPoolExecutor() as executor:
            results = executor.map(extract_events_from_pcap, (str(file) for file in batch_files))
        for events in results:
            for event in events:
                event = flatten_event(event)
                data.append(event)
                labels.append(event.get('label', 0))
        yield data, labels

def main():
    logging.info("Loading and preparing training data...")

    # Process and flatten data
    data_batches, labels_batches = [], []
    for data_batch, labels_batch in process_files_in_batches(INPUT_DIR / "train"):
        data_batches.extend(data_batch)
        labels_batches.extend(labels_batch)

    logging.info("Vectorizing training data using TF-IDF...")
    # Vectorize JSON data with TfidfVectorizer for compactness
    vectorizer = TfidfVectorizer(max_features=500)  # Limit features to top 500 terms
    X_train = vectorizer.fit_transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])

    logging.info("Scaling training data...")
    # Scale data to standardize feature range
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)

    # Optimize Random Forest parameters with limited search space
    model = RandomForestClassifier(random_state=42, warm_start=True, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [20, 50, 100],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    randomized_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=1, random_state=42)
    
    logging.info("Starting hyperparameter search...")
    start_time = time.time()
    randomized_search.fit(X_train, labels_batches)
    best_model = randomized_search.best_estimator_

    logging.info(f"Best model found: {randomized_search.best_params_}")
    logging.info(f"RandomizedSearchCV took {time.time() - start_time:.2f} seconds")

    logging.info("Preparing test data...")
    test_data_batches = []
    for data_batch, _ in process_files_in_batches(INPUT_DIR / 'test', batch_size=20):
        test_data_batches.extend(data_batch)

    X_test = vectorizer.transform([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in test_data_batches])
    X_test = scaler.transform(X_test)

    logging.info("Making predictions on test data...")
    predictions = best_model.predict(X_test)
    
    labels = {file.name: int(pred) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}

    logging.info(f"Writing results to {OUTPUT_DIR / 'labels'}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

    logging.info("Saving model, vectorizer, and metadata...")
    joblib.dump(best_model, OUTPUT_DIR / 'random_forest_model.pkl')
    joblib.dump(vectorizer, OUTPUT_DIR / 'vectorizer.pkl')

    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'model_params': randomized_search.best_params_,
            'feature_names': vectorizer.get_feature_names_out()
        }, f)

if __name__ == "__main__":
    main()
