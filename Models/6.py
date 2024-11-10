# 0.979
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from typing import List, Dict, Tuple
import pickle
from functools import partial
import mmap
import multiprocessing

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
CACHE_DIR = Path('/usr/src/app/cache')
CHUNK_SIZE = 5000  # Increased for better parallelization
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
CACHE_ENABLED = True

class DataProcessor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            max_df=0.85,
            min_df=5,
            stop_words='english',
            dtype=np.float32,
            norm='l2',
            use_idf=True,
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            dtype=np.float32,
            binary=True  # Convert counts to binary features
        )
        self.scaler = StandardScaler(with_mean=False)
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache directory and structures."""
        if CACHE_ENABLED:
            CACHE_DIR.mkdir(exist_ok=True)
            self.cache_file = CACHE_DIR / 'processed_files.pkl'
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.processed_files = pickle.load(f)
            else:
                self.processed_files = set()

    @staticmethod
    def _read_pcap_mmap(file_path: str) -> List[Dict]:
        """Read PCAP file using memory mapping for improved efficiency."""
        events = []
        try:
            with open(file_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                with PcapReader(mm) as pcap:
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
    def flatten_event(event: Dict) -> Dict:
        """Optimized event flattening using dictionary comprehension."""
        return {
            f"{k}_{sk}" if isinstance(v, dict) else k: sv if isinstance(v, dict) else v
            for k, v in event.items()
            for sk, sv in (v.items() if isinstance(v, dict) else [(None, v)])
        }

    def process_file_batch(self, files: List[Path]) -> Tuple[List[Dict], List[int]]:
        """Process a batch of files with improved parallelization."""
        if CACHE_ENABLED:
            # Filter out already processed files
            files_to_process = [f for f in files if f.name not in self.processed_files]
            if not files_to_process:
                return [], []

        data, labels = [], []
        process_func = partial(self._read_pcap_mmap)
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
            futures = [executor.submit(process_func, str(f)) for f in files]
            for future, file in zip(futures, files):
                events = future.result()
                for event in events:
                    flat_event = self.flatten_event(event)
                    data.append(flat_event)
                    labels.append(flat_event.get('label', 0))
                
                if CACHE_ENABLED:
                    self.processed_files.add(file.name)

        return data, labels

    def vectorize_batch(self, data_batch: List[Dict]) -> np.ndarray:
        """Optimized batch vectorization using numpy operations."""
        text_data = [
            " ".join(f"{k}_{v}" for k, v in d.items())
            for d in data_batch
        ]
        
        # Parallel feature extraction
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_tfidf = executor.submit(self.tfidf_vectorizer.transform, text_data)
            future_count = executor.submit(self.count_vectorizer.transform, text_data)
            
            tfidf_features = future_tfidf.result()
            count_features = future_count.result()

        # Use sparse matrix operations
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
    
    # Process files in parallel batches
    all_data = []
    all_labels = []
    
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        # Process chunks in parallel
        chunk_ranges = [(i, min(i + CHUNK_SIZE, len(train_files))) 
                       for i in range(0, len(train_files), CHUNK_SIZE)]
        
        futures = []
        for start, end in chunk_ranges:
            batch_files = train_files[start:end]
            future = executor.submit(data_processor.process_file_batch, batch_files)
            futures.append(future)
        
        # Collect results
        for future in futures:
            data_batch, labels_batch = future.result()
            all_data.extend(data_batch)
            all_labels.extend(labels_batch)
    
    # Fit vectorizers
    logger.info("Vectorizing training data...")
    text_data = [" ".join(f"{k}_{v}" for k, v in d.items()) for d in all_data]
    
    # Parallel fitting of vectorizers
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(data_processor.tfidf_vectorizer.fit, text_data)
        executor.submit(data_processor.count_vectorizer.fit, text_data)
    
    # Process and train
    X_train = data_processor.vectorize_batch(all_data)
    y_train = np.array(all_labels, dtype=np.int32)
    
    # Scale features
    X_train = data_processor.scaler.fit_transform(X_train)
    
    # Initialize and train Random Forest with optimized parameters
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=N_JOBS,
        random_state=42,
        warm_start=True,
        class_weight='balanced'  # Added for better handling of imbalanced classes
    )
    
    model.fit(X_train, y_train)
    
    # Process test data
    logger.info("Processing test data...")
    test_files = list((INPUT_DIR / "test").iterdir())
    test_data = []
    
    # Parallel processing of test data
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        chunk_ranges = [(i, min(i + CHUNK_SIZE, len(test_files))) 
                       for i in range(0, len(test_files), CHUNK_SIZE)]
        
        futures = []
        for start, end in chunk_ranges:
            batch_files = test_files[start:end]
            future = executor.submit(data_processor.process_file_batch, batch_files)
            futures.append(future)
        
        for future in futures:
            data_batch, _ = future.result()
            test_data.extend(data_batch)
    
    # Vectorize and predict test data
    X_test = data_processor.vectorize_batch(test_data)
    X_test = data_processor.scaler.transform(X_test)
    
    # Batch predictions for memory efficiency
    predictions = []
    for i in range(0, len(X_test), CHUNK_SIZE):
        batch = X_test[i:i + CHUNK_SIZE]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    
    # Save results
    logger.info("Saving results...")
    labels = {file.name: int(pred) for file, pred in zip(test_files, predictions)}
    
    with open(OUTPUT_DIR / 'labels', 'w') as f:
        json.dump(labels, f)
    
    # Save cache if enabled
    if CACHE_ENABLED:
        with open(data_processor.cache_file, 'wb') as f:
            pickle.dump(data_processor.processed_files, f)
    
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
