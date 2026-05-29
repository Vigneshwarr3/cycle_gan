#!/usr/bin/env bash

echo "Downloading Berkeley Van Gogh dataset"
wget "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip"
unzip -q vangogh2photo.zip -d dataset
echo "Download completed!"