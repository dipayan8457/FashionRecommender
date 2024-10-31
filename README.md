# Fashion Recommender System

This project is a Fashion Recommender System that uses deep learning and computer vision techniques to suggest visually similar fashion items based on an uploaded image. Built using a pre-trained ResNet50 model for feature extraction and a k-nearest neighbors (k-NN) algorithm for finding similar items, this system provides an interactive web app interface via Streamlit and a local test script using OpenCV.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Overview

The Fashion Recommender System allows users to upload an image of a fashion item and receive recommendations of similar items based on visual similarity. The system extracts features from images using a pre-trained ResNet50 model (without the top layers), which are then used to identify similar images through k-NN search in the embedding space.

## Features

- **Image Feature Extraction**: Uses ResNet50 to convert images into feature vectors.
- **Similarity Search**: Finds visually similar images using k-NN with Euclidean distance.
- **Web Interface**: Streamlit app allows users to upload an image and view recommendations.
- **Testing Script**: Command-line script (`test.py`) displays similar images using OpenCV.

## Dependencies

- Python 3.6+
- TensorFlow
- Streamlit
- OpenCV
- Scikit-learn
- NumPy
- tqdm
- PIL

## Acknowledgments

This project utilizes:
- **ResNet50** from TensorFlow for feature extraction.
- **k-Nearest Neighbors** from Scikit-learn for similarity search.
- **Streamlit** for building an interactive web interface.
  
---
