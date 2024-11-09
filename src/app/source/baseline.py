import json
from pathlib import Path
from scapy.all import rdpcap, TCP, IP
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict

INPUT_DIR = Path('/mnt/c/users/ionut/desktop/usr/src/app/InputData')
OUTPUT_DIR = Path('/mnt/c/users/ionut/desktop/usr/src/app/output')

def extract_events_from_pcap(file_path):
    # Extract JSON events from TCP payloads in PCAP file
    packets = rdpcap(file_path)
    events = []
    
    for pkt in packets:
        if pkt.haslayer('TCP') and pkt['TCP'].payload:
            try:
                payload = pkt['TCP'].payload.load.decode('utf-8')  # Decode the payload to string
                
                # Check if the payload is already a dictionary
                try:
                    event = json.loads(payload)  # Try to parse as JSON
                except json.JSONDecodeError:
                    event = payload  # If not a valid JSON, keep it as a string
                
                # If it's not a string (already a dictionary), append it directly
                if isinstance(event, dict):
                    events.append(event)
                else:
                    # Handle cases where event is not a dictionary
                    continue
                    
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                # Skip packets with decoding errors
                continue
    
    return events



def flatten_dict(d, parent_key='', sep='_'):
    # Helper function to flatten a dictionary
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If the value is a list, join it into a string
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

def preprocess_event(event):
    # Preprocess each event dictionary to handle nested structures
    return flatten_dict(event)

def prepare_data(input_dir, is_training=True):
    # Load and process data from PCAP files
    data, labels = [], []
    for file in input_dir.iterdir():
        events = extract_events_from_pcap(str(file))

        for event in events:
            event = preprocess_event(event)  # Flatten the event dictionary
            if is_training:
                labels.append(event.get('label'))
            data.append(event)
    return data, labels if is_training else data

def main():
    # Prepare the training data
    train_data, train_labels = prepare_data(INPUT_DIR / "train")
    
    # Vectorize the JSON data to prepare for the Isolation Forest
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(train_data)
    
    # Train the Isolation Forest model
    model = IsolationForest(contamination=0.2, random_state=42)
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
