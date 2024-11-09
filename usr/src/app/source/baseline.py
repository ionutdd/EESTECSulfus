import json
import logging
from pathlib import Path
from scapy.all import rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import joblib
import os
import time

# Set up logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')

def extract_events_from_pcap(file_path):
    """Efficiently extract events from PCAP files."""
    packets = rdpcap(file_path)
    events = []

    # Iterate over packets and extract the JSON data from TCP payload
    for pkt in packets:
        if pkt.haslayer('TCP') and pkt['TCP'].payload:
            try:
                payload = pkt['TCP'].payload.load.decode('utf-8')  # Decode the payload to string
                try:
                    event = json.loads(payload)  # Try to parse as JSON
                    if isinstance(event, dict):  # If valid JSON dictionary
                        events.append(event)
                except json.JSONDecodeError:
                    continue  # Ignore non-JSON payloads
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # Skip packets with decoding errors

    return events

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionaries into a single-level dictionary with composite keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(map(str, v))))  # Convert lists to string
        else:
            items.append((new_key, v))
    return dict(items)

def preprocess_event(event):
    """Preprocess each event by flattening nested structures."""
    return flatten_dict(event)

def process_files_in_batches(input_dir, batch_size=10):
    """Process files in batches to reduce memory consumption."""
    files = list(input_dir.iterdir())
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        data, labels = [], []
        with ProcessPoolExecutor() as executor:
            results = executor.map(extract_events_from_pcap, (str(file) for file in batch_files))
        for events in results:
            for event in events:
                event = preprocess_event(event)
                data.append(event)
                labels.append(event.get('label', 0))  # Default to 0 if 'label' is missing
        yield data, labels

def main():
    logging.info("Loading and preparing training data...")

    # Prepare the training data
    data_batches, labels_batches = [], []
    for data_batch, labels_batch in process_files_in_batches(INPUT_DIR / "train"):
        data_batches.extend(data_batch)
        labels_batches.extend(labels_batch)

    logging.info("Vectorizing training data...")
    # Vectorize the JSON data
    vectorizer = DictVectorizer(sparse=True)  # Sparse format for efficiency
    X_train = vectorizer.fit_transform(data_batches)

    logging.info("Scaling training data...")
    # Scale the data (with_mean=False to optimize memory for sparse data)
    scaler = StandardScaler(with_mean=False)  # Don't scale to zero mean to keep efficiency with sparse matrices
    X_train = scaler.fit_transform(X_train)

    # Use Random Forest for classification with hyperparameter tuning
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': np.arange(50, 301, 50),  # Number of trees in the forest
        'max_depth': [None, 10, 50, 100],  # Max depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4, 8],  # Minimum number of samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider at each split
    }

    # RandomizedSearchCV for hyperparameter tuning
    randomized_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                           n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
    
    logging.info("Starting hyperparameter search...")
    start_time = time.time()
    randomized_search.fit(X_train, labels_batches)
    best_model = randomized_search.best_estimator_
    
    logging.info(f"Best model found: {randomized_search.best_params_}")
    logging.info(f"RandomizedSearchCV took {time.time() - start_time:.2f} seconds")

    # Prepare and predict on test data
    logging.info("Preparing test data...")
    test_data_batches = []
    for data_batch, _ in process_files_in_batches(INPUT_DIR / 'test', batch_size=10):
        test_data_batches.extend(data_batch)
    
    X_test = vectorizer.transform(test_data_batches)
    X_test = scaler.transform(X_test)

    logging.info("Making predictions on test data...")
    # Predict labels using the trained Random Forest model
    predictions = best_model.predict(X_test)

    # Convert predictions: 1 for normal, 0 for anomalous (or class-specific labels)
    labels = {file.name: int(pred) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}

    logging.info(f"Writing results to {OUTPUT_DIR / 'labels'}...")
    # Output results to the specified directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

    # Save the model, vectorizer, and metadata to disk for future use
    logging.info("Saving model, vectorizer, and metadata...")
    joblib.dump(best_model, OUTPUT_DIR / 'random_forest_model.pkl')
    joblib.dump(vectorizer, OUTPUT_DIR / 'vectorizer.pkl')

    # Optionally save metadata (like hyperparameters and feature names)
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'model_params': randomized_search.best_params_,
            'model_score': best_model.score(X_test, labels),  # Add evaluation score
            'feature_names': vectorizer.get_feature_names_out()
        }, f)

if __name__ == "__main__":
    main()
