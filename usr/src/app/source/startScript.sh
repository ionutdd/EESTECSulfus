#!/bin/bash

# Activate the virtual environment
source usr/src/app/venv/bin/activate

# Run the provided Python script
echo "Running baseline.py"
python3.9 usr/src/app/source/baseline.py

echo "Script execution completed."

# Deactivate the virtual environment
deactivate
