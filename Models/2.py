#0.971

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
    logging.info("Loading and preparing training data...")

    # Process and flatten data
    data_batches, labels_batches = [], []
    for data_batch, labels_batch in process_files_in_batches(INPUT_DIR / "train"):
        data_batches.extend(data_batch)
        labels_batches.extend(labels_batch)

    logging.info("Vectorizing training data using TF-IDF + CountVectorizer...")
    # Vectorize JSON data with TF-IDF and CountVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, max_df=0.85, min_df=5, stop_words='english')
    count_vectorizer = CountVectorizer(max_features=500, stop_words='english')

    # Fit the vectorizers on the training data
    tfidf_vectorizer.fit([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])
    count_vectorizer.fit([" ".join([f"{k}_{v}" for k, v in d.items()]) for d in data_batches])

    tfidf_X_train, count_X_train = vectorize_data(data_batches, tfidf_vectorizer, count_vectorizer)

    if tfidf_X_train is None or count_X_train is None:
        logging.error("Error during vectorization, aborting...")
        return

    # Combine both features
    X_train_combined = np.hstack([tfidf_X_train.toarray(), count_X_train.toarray()])

    logging.info("Scaling training data...")
    # Scale data to standardize feature range (only if necessary)
    scaler = StandardScaler(with_mean=False)
    X_train_combined = scaler.fit_transform(X_train_combined)

    # Optimize Random Forest parameters with a more focused search
    model = RandomForestClassifier(random_state=42, warm_start=True, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [20, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Use StratifiedKFold for better cross-validation to handle imbalanced classes
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    randomized_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=cv, n_jobs=-1, verbose=1, random_state=42)

    logging.info("Starting hyperparameter search...")
    start_time = time.time()
    randomized_search.fit(X_train_combined, labels_batches)
    best_model = randomized_search.best_estimator_

    logging.info(f"Best model found: {randomized_search.best_params_}")
    logging.info(f"RandomizedSearchCV took {time.time() - start_time:.2f} seconds")

    logging.info("Preparing test data...")
    test_data_batches = []
    for data_batch, _ in process_files_in_batches(INPUT_DIR / 'test', batch_size=20):
        test_data_batches.extend(data_batch)

    tfidf_X_test, count_X_test = vectorize_data(test_data_batches, tfidf_vectorizer, count_vectorizer)

    if tfidf_X_test is None or count_X_test is None:
        logging.error("Error during test data vectorization, aborting...")
        return

    # Combine test features
    X_test_combined = np.hstack([tfidf_X_test.toarray(), count_X_test.toarray()])
    X_test_combined = scaler.transform(X_test_combined)

    logging.info("Making predictions on test data...")
    predictions = best_model.predict(X_test_combined)
    
    labels = {file.name: int(pred) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}

    logging.info(f"Writing results to {OUTPUT_DIR / 'labels'}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

    logging.info("Saving model, vectorizer, and metadata...")
    joblib.dump(best_model, OUTPUT_DIR / 'random_forest_model.pkl')
    joblib.dump(tfidf_vectorizer, OUTPUT_DIR / 'tfidf_vectorizer.pkl')
    joblib.dump(count_vectorizer, OUTPUT_DIR / 'count_vectorizer.pkl')

    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'model_params': randomized_search.best_params_,
            'tfidf_feature_names': tfidf_vectorizer.get_feature_names_out(),
            'count_feature_names': count_vectorizer.get_feature_names_out()
        }, f)

if __name__ == "__main__":
    main()
