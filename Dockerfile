# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the entire project into the container (excluding unnecessary files via .dockerignore)
COPY . .

# Debugging step: List all files in /usr/src/app to verify that DataFolder is copied
RUN echo "Listing all files in /usr/src/app"
RUN ls -R usr/src/app

# Debugging step: Check specifically the contents of /usr/src/app/DataFolder
RUN echo "Listing contents of /usr/src/app/DataFolder"
RUN ls -R usr/src/app/DataFolder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    unzip \
    && rm -rf var/lib/apt/lists/*

# Install pip and virtual environment tools
RUN pip install --upgrade pip
RUN pip install virtualenv

# Create and activate a Python 3.9 virtual environment
RUN python3.9 -m venv usr/src/app/venv
RUN usr/src/app/venv/bin/pip install --upgrade pip

# Install dependencies using the provided packageScript.sh
RUN bash usr/src/app/DataFolder/packageScript.sh

# Install Python packages from requirements.txt inside the virtual environment
RUN usr/src/app/venv/bin/pip install -r usr/src/app/DataFolder/requirements.txt

# Ensure startScript.sh has execution permissions
RUN chmod +x usr/src/app/source/startScript.sh

# Set the virtual environment path
ENV PATH="usr/src/app/venv/bin:$PATH"

# Set the default command to run the project
CMD ["/bin/bash", "usr/src/app/source/startScript.sh"]
