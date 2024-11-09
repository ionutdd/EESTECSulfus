import json
import logging
from pathlib import Path
from scapy.all import rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
import os

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

def prepare_data(input_dir, is_training=True):
    """Prepare data from PCAP files for model training and testing."""
    data, labels = [], []
    logging.info(f"Processing files in {input_dir}...")

    # Use ProcessPoolExecutor to parallelize the extraction of data
    with ProcessPoolExecutor() as executor:
        results = executor.map(extract_events_from_pcap, (str(file) for file in input_dir.iterdir()))

    # Flatten events and accumulate the data
    for events in results:
        for event in events:
            event = preprocess_event(event)  # Flatten the event dictionary
            data.append(event)
            if is_training:
                labels.append(event.get('label', 0))  # Default to 0 if 'label' is missing
    return data, labels if is_training else data

def main():
    logging.info("Loading and preparing training data...")
    # Prepare the training data
    train_data, train_labels = prepare_data(INPUT_DIR / "train")
    
    logging.info("Vectorizing training data...")
    # Vectorize the JSON data
    vectorizer = DictVectorizer(sparse=True)  # Sparse format for efficiency
    X_train = vectorizer.fit_transform(train_data)
    
    logging.info("Scaling training data...")
    # Scale the data (with_mean=False to optimize memory for sparse data)
    scaler = StandardScaler(with_mean=False)  # Don't scale to zero mean to keep efficiency with sparse matrices
    X_train = scaler.fit_transform(X_train)
    
    # Use Random Forest for classification with hyperparameter tuning
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    logging.info("Starting hyperparameter search...")
    grid_search.fit(X_train, train_labels)
    best_model = grid_search.best_estimator_
    
    logging.info(f"Best model found: {grid_search.best_params_}")
    
    # Prepare and predict on test data
    logging.info("Preparing test data...")
    test_data, _ = prepare_data(INPUT_DIR / 'test', is_training=False)
    X_test = vectorizer.transform(test_data)
    X_test = scaler.transform(X_test)
    
    logging.info("Making predictions on test data...")
    # Predict labels using the trained Random Forest model
    predictions = best_model.predict(X_test)
    
    # Convert predictions: 1 for normal, 0 for anomalous (or class-specific labels)
    labels = {file.name: int(pred) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}
    
    logging.info(f"Writing results to {OUTPUT_DIR / 'labels'}...")
    # Output results to the specified directory
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

    # Save the model, vectorizer, and metadata to disk for future use
    logging.info("Saving model and vectorizer...")
    joblib.dump(best_model, OUTPUT_DIR / 'random_forest_model.pkl')
    joblib.dump(vectorizer, OUTPUT_DIR / 'vectorizer.pkl')

    # Optionally save metadata (like hyperparameters and feature names)
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'model_params': grid_search.best_params_,
            'feature_names': vectorizer.get_feature_names_out()
        }, f)

if __name__ == "__main__":
    main()
