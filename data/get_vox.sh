#!/usr/bin/env bash
wget www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/dense-face-frames.tar.gz
echo "Extracting..."
tar -xf dense-face-frames.tar.gz
rm -rf dense-face-frames.tar.gz
echo "Converting..."
python preprocess_vox.py
rm -rf unzippedIntervalFaces
