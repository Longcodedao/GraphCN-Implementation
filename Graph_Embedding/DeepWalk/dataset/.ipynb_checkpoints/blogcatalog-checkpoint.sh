#!/bin/bash

# Install unzip if not already installed
sudo apt-get install -y unzip

# Download the BlogCatalog dataset
wget --no-check-certificate "https://datasets.syr.edu/uploads/1283153973/BlogCatalog-dataset.zip"

# Unzip the downloaded file
unzip BlogCatalog-dataset.zip
