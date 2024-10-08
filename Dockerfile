# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Install Git and Git-Annex along with their dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-annex \
    datalad \
    && rm -rf /var/lib/apt/lists/*

# Set Git user name and email globally
RUN git config --global user.name "Bhanu Prasanna" \
    && git config --global user.email "bhanu.prasanna2001@gmail.com"


# Install Python packages using pip
RUN pip install --no-cache-dir \
    scikit-learn \
    datalad \
    datalad-container \
    pandas \
    joblib \
    matplotlib
