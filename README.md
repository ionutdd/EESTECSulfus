# EESTEC Challenge - Team Sulfus

# Windows Log Classifier
A machine learning pipeline for processing PCAP files, extracting and vectorizing events, and training a Random Forest model to classify network events with high precision. This project was developed as part of a **24-hour hackathon**, achieving a **precision of 98%** which helped us win **3rd place**. [View the PowerPoint Presentation](https://www.canva.com/design/DAGVWQg2nyc/S7or-5Hfy90PsidPC--Z0w/view?utm_content=DAGVWQg2nyc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## Project Overview
This project uses PCAP (packet capture) files as input, extracts relevant network events, and applies feature engineering and machine learning techniques to classify the events. Our model, based on a **Random Forest classifier**, leverages concurrent processing, feature vectorization, and hyperparameter tuning to reach high accuracy and precision. The code is organized to handle real-world datasets with efficient memory usage and parallelism.

## Install the dependencies with:

``./packageScript.sh``

## Features

**PCAP Parsing with Scapy**: Extracts TCP payload data from PCAP files for further processing. <br />
**Concurrent File Processing**: Uses Python's ProcessPoolExecutor for batch processing and concurrency, enabling the handling of large datasets efficiently. <br />
**Feature Engineering**: Leverages TF-IDF and CountVectorizer for numerical feature representation of text data. <br />
**Model Training with Random Forest**: Trains a Random Forest classifier with hyperparameter tuning using RandomizedSearchCV for optimal performance. <br />
**High Precision**: Achieved a precision of 98%, demonstrating robust performance in network event classification. <br />

## Project Workflow
**Data Extraction**: Parses PCAP files to decode TCP payloads and extract JSON-encoded event data. <br />
**Batch Processing**: Splits data into manageable batches and processes them in parallel, which is ideal for handling large data volumes. <br />
**Vectorization**: Converts extracted text data into numerical features using TF-IDF and Count Vectorizer. <br />
**Model Training**: Uses a Random Forest classifier with hyperparameter tuning to optimize accuracy and precision. <br />
**Prediction and Output**: After training, the model predicts classifications on test data and saves the results. <br />

## File Structure

├── InputData            # Folder for input PCAP files <br />
├── output               # Folder for model and prediction outputs <br />
├── baseline.py          # Main script for data processing, model training, and prediction <br />
├── packageScript.sh     # Project dependencies <br />

## Usage

**Prepare Data**: Place the training PCAP files in the InputData/train directory and testing files in InputData/test. <br />
**Run the Pipeline**: Execute main.py to process the data, train the model, and make predictions. <br />
**View Results**: Predictions and model files will be saved in the output directory. <br />

## Hackathon Highlights

This project was completed in a high-pressure, 24-hour hackathon setting, emphasizing quick problem-solving and efficient code organization. Despite the time constraints, we achieved high model precision (98%), highlighting the effectiveness of ensemble learning and optimized processing.

## Results

**Precision**: 98% <br />
**Optimization**: Efficient batch processing and memory management allowed for real-time data handling. <br />
**High Dimensionality Handling**: The Random Forest model and vectorization techniques performed well on high-dimensional data, showcasing the project's scalability.
