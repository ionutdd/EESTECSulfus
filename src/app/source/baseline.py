import json
import os
from pathlib import Path
from scapy.all import rdpcap, TCP, IP
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
import time
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_DIR = Path('src/app/InputData')
OUTPUT_DIR = Path('src/app/output')
DIRECTORIES = {
    'raw_train': OUTPUT_DIR / 'raw_train',
    'raw_test': OUTPUT_DIR / 'raw_test',
    'processed_train': OUTPUT_DIR / 'processed_train',
    'processed_test': OUTPUT_DIR / 'processed_test',
    'raw': OUTPUT_DIR / 'raw'
}

def setup_directories():
    """Clear and recreate all directories."""
    for directory in DIRECTORIES.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

def extract_events_from_pcap(file_path):
    """Extract events from PCAP file and save raw data."""
    events = []
    non_json_payloads = []
    non_dict_events = []
    
    logger.info(f"Processing file: {file_path}")
    
    try:
        packets = rdpcap(str(file_path))
        logger.info(f"Number of packets in {file_path}: {len(packets)}")
        
        for pkt in packets:
            if pkt.haslayer(TCP) and pkt[TCP].payload:
                try:
                    payload = bytes(pkt[TCP].payload)
                    try:
                        decoded_payload = payload.decode('utf-8')
                        try:
                            event = json.loads(decoded_payload)
                            if isinstance(event, dict):
                                events.append(event)
                            else:
                                non_dict_events.append(decoded_payload)
                        except json.JSONDecodeError:
                            non_json_payloads.append(decoded_payload)
                    except UnicodeDecodeError:
                        continue
                except Exception as e:
                    logger.error(f"Error processing packet: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error reading PCAP file {file_path}: {e}")
        return []

    logger.info(f"Extracted {len(events)} valid events from {file_path}")
    return events

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = ', '.join(map(str, v))
        else:
            items[new_key] = v
    return items

def process_file(file_path, is_training=True):
    """Process a single PCAP file and save both raw and processed data."""
    file_path = Path(file_path)
    logger.info(f"Processing file: {file_path}")
    
    # Extract events
    events = extract_events_from_pcap(file_path)
    
    # Save raw data
    raw_dir = DIRECTORIES['raw_train'] if is_training else DIRECTORIES['raw_test']
    raw_output_path = raw_dir / f"raw_{file_path.name}.json"
    with open(raw_output_path, 'w') as f:
        json.dump(events, f, indent=4)
    
    # Process events
    processed_events = []
    labels = []
    
    for event in events:
        processed_event = flatten_dict(event)
        if is_training and 'label' in processed_event:
            labels.append(processed_event.pop('label'))
        processed_events.append(processed_event)
    
    # Save processed data
    processed_dir = DIRECTORIES['processed_train'] if is_training else DIRECTORIES['processed_test']
    processed_output_path = processed_dir / f"processed_{file_path.name}.json"
    with open(processed_output_path, 'w') as f:
        json.dump(processed_events, f, indent=4)
    
    return processed_events, labels

def prepare_data(input_dir, is_training=True):
    """Prepare data from PCAP files."""
    input_dir = Path(input_dir)
    files = [f for f in input_dir.iterdir() if f.is_file()]
    logger.info(f"Found {len(files)} files in {input_dir}")
    
    all_data = []
    all_labels = []
    
    for file in files:
        try:
            data, labels = process_file(file, is_training)
            all_data.extend(data)
            if is_training:
                all_labels.extend(labels)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid data extracted from {input_dir}")
        
    return all_data, all_labels if is_training else (all_data, None)

def main():
    start_time = time.time()
    
    # Setup directories
    setup_directories()
    
    try:
        # Training phase
        logger.info("Starting training data preparation...")
        train_data, train_labels = prepare_data(INPUT_DIR / "train", True)
        
        if not train_data:
            raise ValueError("No training data available")
        
        logger.info(f"Number of training samples: {len(train_data)}")
        
        vectorizer = DictVectorizer(sparse=False)
        X_train = vectorizer.fit_transform(train_data)
        
        logger.info("Training Isolation Forest model...")
        model = IsolationForest(contamination=0.49, random_state=42)
        model.fit(X_train)
        
        # Testing phase
        logger.info("Starting test data preparation...")
        test_data, _ = prepare_data(INPUT_DIR / 'test', False)
        
        if not test_data:
            raise ValueError("No test data available")
            
        logger.info(f"Number of test samples: {len(test_data)}")
        
        X_test = vectorizer.transform(test_data)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Generate labels for test files
        test_files = sorted([f for f in (INPUT_DIR / 'test').iterdir() if f.is_file()])
        
        # Create labels dictionary
        labels = {}
        current_idx = 0
        
        for file in test_files:
            processed_file = DIRECTORIES['processed_test'] / f"processed_{file.name}.json"
            with open(processed_file, 'r') as f:
                file_events = json.load(f)
            
            file_predictions = predictions[current_idx:current_idx + len(file_events)]
            current_idx += len(file_events)
            
            # File is anomalous if any event is anomalous
            is_anomalous = any(pred == -1 for pred in file_predictions)
            labels[file.name] = 1 if is_anomalous else 0
        
        # Save results
        filtered_labels = {k: v for k, v in labels.items() if v == 1}
        
        outputs = {
            'labels': filtered_labels,
            'labels.json': labels,
            'test_predictions.json': labels,
            'train_predictions.json': train_labels
        }
        
        for filename, content in outputs.items():
            with open(OUTPUT_DIR / filename, 'w') as f:
                json.dump(content, f, indent=4)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total running time: {elapsed_time:.2f} seconds")
    logger.info(f"Anomalous files detected: {sum(1 for v in labels.values() if v == 1)}")

if __name__ == "__main__":
    main()