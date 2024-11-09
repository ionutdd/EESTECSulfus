import json
import random
from pathlib import Path
from scapy.all import rdpcap
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer

INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')

def extract_events_from_pcap(file_path):
    # Extract JSON events from TCP payloads in PCAP file
    packets = rdpcap(file_path)
    events = []
    for pkt in packets:
        if pkt.haslayer('TCP') and pkt['TCP'].payload:
            try:
                payload = pkt['TCP'].payload.load.decode('utf-8')
                event = json.loads(payload)
                events.append(event)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return events

def prepare_data(input_dir, is_training=True):
    # Load and process data from PCAP files
    data, labels = [], []
    for file in input_dir.iterdir():
        events = extract_events_from_pcap(file)
        for event in events:
            if is_training:
                labels.append(event.get("label", 0))
            data.append(event)
    return data, labels if is_training else data

def main():
    # Prepare the training data
    train_data, train_labels = prepare_data(INPUT_DIR / "train")
    
    # Vectorize the JSON data to prepare for the Isolation Forest
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(train_data)
    
    # Train the Isolation Forest model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    
    # Prepare and predict on test data
    test_data, _ = prepare_data(INPUT_DIR / 'test', is_training=False)
    X_test = vectorizer.transform(test_data)
    predictions = model.predict(X_test)
    
    # Output results
    labels = {file.name: int(pred == -1) for file, pred in zip((INPUT_DIR / "test").iterdir(), predictions)}
    (OUTPUT_DIR / 'labels').write_text(json.dumps(labels))

if __name__ == "__main__":
    main()
