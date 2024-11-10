# 0.978

import json
import logging
import time
from pathlib import Path
import numpy as np
import joblib
import os
from scapy.all import PcapReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')
CHUNK_SIZE = 1000
N_JOBS = os.cpu_count() or 4

class DataProcessor:
    def __init__(self):
        # Optimized vectorizer parameters based on the original high-performing setup
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            max_df=0.85,
            min_df=5,
            stop_words='english',
            dtype=np.float32
        )
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            dtype=np.float32
        )
        self.scaler = StandardScaler(with_mean=False)
        
    @staticmethod
    def extract_events_from_pcap(file_path: str):
        """Extract events from PCAP files using PcapReader for memory efficiency."""
        events = []
        try:
            with PcapReader(file_path) as pcap:
                for pkt in pcap:
                    if pkt.haslayer('TCP') and pkt['TCP'].payload:
                        try:
                            payload = bytes(pkt['TCP'].payload)
                            event = json.loads(payload)
                            if isinstance(event, dict):
                                events.append(event)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        return events

    @staticmethod
    def flatten_event(event):
        """Flatten nested structures efficiently."""
        flattened = {}
        for k, v in event.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flattened[f"{k}_{sk}"] = sv
            else:
                flattened[k] = v
        return flattened

    def process_file_batch(self, files):
        """Process a batch of files with parallel execution."""
        data, labels = [], []
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            for events in executor.map(self.extract_events_from_pcap, (str(f) for f in files)):
                for event in events:
                    flat_event = self.flatten_event(event)
                    data.append(flat_event)
                    labels.append(flat_event.get('label', 0))
        return data, labels

    def vectorize_batch(self, data_batch):
        """Vectorize a batch of data efficiently."""
        text_data = [" ".join(f"{k}_{v}" for k, v in d.items()) for d in data_batch]
        tfidf_features = self.tfidf_vectorizer.transform(text_data)
        count_features = self.count_vectorizer.transform(text_data)
        return np.hstack([tfidf_features.toarray(), count_features.toarray()])

def main():
    start_time = time.time()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize processor
    data_processor = DataProcessor()
    
    # Process training data
    logger.info("Processing training data...")
    train_files = list((INPUT_DIR / "train").iterdir())
    
    # Process files in batches
    all_data = []
    all_labels = []
    for i in range(0, len(train_files), CHUNK_SIZE):
        batch_files = train_files[i:i + CHUNK_SIZE]
        data_batch, labels_batch = data_processor.process_file_batch(batch_files)
        all_data.extend(data_batch)
        all_labels.extend(labels_batch)
    
    # Fit vectorizers
    logger.info("Vectorizing training data...")
    text_data = [" ".join(f"{k}_{v}" for k, v in d.items()) for d in all_data]
    data_processor.tfidf_vectorizer.fit(text_data)
    data_processor.count_vectorizer.fit(text_data)
    
    # Process and train
    X_train = data_processor.vectorize_batch(all_data)
    y_train = np.array(all_labels)
    
    # Scale features
    X_train = data_processor.scaler.fit_transform(X_train)
    
    # Initialize and train Random Forest with optimized parameters
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,  # Allow full depth for better accuracy
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',  # Use sqrt(n_features) features at each split
        bootstrap=True,
        n_jobs=N_JOBS,
        random_state=42,
        warm_start=True  # Enable warm start for faster training
    )
    
    model.fit(X_train, y_train)
    
    # Process test data
    logger.info("Processing test data...")
    test_files = list((INPUT_DIR / "test").iterdir())
    test_data = []
    for i in range(0, len(test_files), CHUNK_SIZE):
        batch_files = test_files[i:i + CHUNK_SIZE]
        data_batch, _ = data_processor.process_file_batch(batch_files)
        test_data.extend(data_batch)
    
    # Vectorize and predict test data
    X_test = data_processor.vectorize_batch(test_data)
    X_test = data_processor.scaler.transform(X_test)
    predictions = model.predict(X_test)
    
    # Save results
    logger.info("Saving results...")
    labels = {file.name: int(pred) for file, pred in zip(test_files, predictions)}
    
    with open(OUTPUT_DIR / 'labels', 'w') as f:
        json.dump(labels, f)
    
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
  
