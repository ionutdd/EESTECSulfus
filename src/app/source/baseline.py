import json
import os
from pathlib import Path
from scapy.all import rdpcap, TCP, IP
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
import time
from concurrent.futures import ProcessPoolExecutor

INPUT_DIR = Path('src/app/InputData')
OUTPUT_DIR = Path('src/app/output')

# Create output directories upfront
raw_train_dir = OUTPUT_DIR / 'raw_train'
raw_test_dir = OUTPUT_DIR / 'raw_test'
processed_train_dir = OUTPUT_DIR / 'processed_train'
processed_test_dir = OUTPUT_DIR / 'processed_test'

# Ensure all output directories exist upfront
raw_train_dir.mkdir(parents=True, exist_ok=True)
raw_test_dir.mkdir(parents=True, exist_ok=True)
processed_train_dir.mkdir(parents=True, exist_ok=True)
processed_test_dir.mkdir(parents=True, exist_ok=True)

def extract_events_from_pcap(file_path):
    # Extract JSON events from TCP payloads in PCAP file
    packets = rdpcap(file_path)
    events = []
    non_json_payloads = []  # To store non-JSON payloads for analysis
    non_dict_events = []    # To store non-dictionary events for analysis

    print(f"Processing file: {file_path}, number of packets: {len(packets)}")

    for pkt in packets:
        if pkt.haslayer('TCP') and pkt['TCP'].payload:
            try:
                payload = pkt['TCP'].payload.load.decode('utf-8')  # Decode the payload to string
                try:
                    event = json.loads(payload)  # Try to parse as JSON
                    if isinstance(event, dict):
                        events.append(event)
                    else:
                        non_dict_events.append(event)
                except json.JSONDecodeError:
                    non_json_payloads.append(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                continue

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if non_json_payloads:
        with open(raw_train_dir / f'non_json_payloads_{timestamp}.json', 'w') as f:
            json.dump(non_json_payloads, f, indent=4)

    if non_dict_events:
        with open(raw_train_dir / f'non_dict_events_{timestamp}.json', 'w') as f:
            json.dump(non_dict_events, f, indent=4)

    return events

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def preprocess_event(event):
    return flatten_dict(event)

def prepare_data(input_dir, is_training=True):
    data, labels = [], []
    files = list(input_dir.iterdir())
    print(f"Preparing {'training' if is_training else 'test'} data from: {input_dir}")

    # Use ProcessPoolExecutor to parallelize file processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_events_from_pcap, [str(file) for file in files]))

    for file, events in zip(files, results):
        print(f"Processing file: {file.name}, events found: {len(events)}")

        for event in events:
            event = preprocess_event(event)  # Flatten the event dictionary
            if is_training:
                labels.append(event.get('label'))
            data.append(event)

        # Batching: Only create directories and write data once at the end
        raw_dir = raw_train_dir if is_training else raw_test_dir
        processed_dir = processed_train_dir if is_training else processed_test_dir
        
        # Save raw events in batches
        with open(raw_dir / f"raw_{file.name}.json", 'w') as f:
            json.dump(events, f, indent=4)

        if data:
            with open(processed_dir / f"processed_{file.name}.json", 'w') as f:
                json.dump(data, f, indent=4)

    print(f"{'Training' if is_training else 'Test'} data prepared: {len(data)} records")
    return data, labels if is_training else data

def load_processed_data(processed_dir):
    data, labels = [], []
    files = list(processed_dir.iterdir())
    for file in files:
        with open(file, 'r') as f:
            events = json.load(f)
            for event in events:
                if 'label' in event:
                    labels.append(event.pop('label'))  # Extract label if it exists
                data.append(event)
    return data, labels

def main():
    start_time = time.time()

    # Prepare training data
    train_data, train_labels = prepare_data(INPUT_DIR / "train")
    train_data, train_labels = load_processed_data(processed_train_dir)

    # Vectorize the data
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(train_data)

    # Train the IsolationForest model
    model = IsolationForest(contamination=0.49, random_state=42)
    model.fit(X_train)

    with open(OUTPUT_DIR / "train_predictions.json", 'w') as f:
        json.dump(train_labels, f, indent=4)

    # Prepare test data
    test_data, _ = prepare_data(INPUT_DIR / 'test', is_training=False)
    test_data, _ = load_processed_data(processed_test_dir)
    X_test = vectorizer.transform(test_data)

    print(f"Training data size: {X_train.shape}")
    print(f"Test data size: {X_test.shape[0]}")

    predictions = model.predict(X_test)

    labels = {}
    test_files = list(processed_test_dir.iterdir())
    for file, pred in zip(test_files, predictions):
        labels[file.name] = 1 if pred == -1 else 0

    filtered_labels = {file.replace("processed_", "").replace(".json", ""): label
                       for file, label in labels.items() if label == 1}

    with open(OUTPUT_DIR / 'labels', 'w') as f:
        json.dump(filtered_labels, f, indent=4)

    with open(OUTPUT_DIR / 'labels.json', 'w') as f:
        json.dump(labels, f, indent=4)

    with open(OUTPUT_DIR / 'test_predictions.json', 'w') as f:
        json.dump(labels, f, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
