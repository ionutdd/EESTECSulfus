#!/bin/bash

# Activate the virtual environment
source /mnt/c/users/ionut/desktop/usr/src/app/source/venv/bin/activate

# Run the provided Python script
echo "Running baseline.py"
python /mnt/c/users/ionut/desktop/usr/src/app/source/baseline.py

echo "Script execution completed."

# Deactivate the virtual environment
deactivate