#!/bin/bash
# Unzip and install packages
unzip -o /usr/src/app/DataFolder/dependencies.zip -d /usr/src/app/DataFolder
pip install --no-index --find-links=/usr/src/app/DataFolder scapy scikit-learn
