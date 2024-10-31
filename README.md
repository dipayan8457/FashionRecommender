# Fashion Recommender System

This project is a Fashion Recommender System that uses deep learning and computer vision techniques to suggest visually similar fashion items based on an uploaded image. Built using a pre-trained ResNet50 model for feature extraction and a k-nearest neighbors (k-NN) algorithm for finding similar items, this system provides an interactive web app interface via Streamlit and a local test script using OpenCV.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Overview

The Fashion Recommender System allows users to upload an image of a fashion item and receive recommendations of similar items based on visual similarity. The system extracts features from images using a pre-trained ResNet50 model (without the top layers), which are then used to identify similar images through k-NN search in the embedding space.

## Features

- **Image Feature Extraction**: Uses ResNet50 to convert images into feature vectors.
- **Similarity Search**: Finds visually similar images using k-NN with Euclidean distance.
- **Web Interface**: Streamlit app allows users to upload an image and view recommendations.
- **Testing Script**: Command-line script (`test.py`) displays similar images using OpenCV.

## Project Structure

```plaintext
.
├── app.py             # Generates feature embeddings for all images in the dataset
├── main.py            # Streamlit app to upload images and display recommendations
├── test.py            # Command-line script to test similarity search using a sample image
├── embeddings.pkl     # Pickle file containing precomputed feature vectors
├── filenames.pkl      # Pickle file containing filenames corresponding to feature vectors
├── images/            # Directory containing dataset images
├── uploads/           # Directory for storing user-uploaded images
└── README.md          # Project documentation
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fashion-recommender.git
   cd fashion-recommender
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**:
   - Place your images in the `images/` directory. These images will serve as the dataset for similarity search.

4. **Generate Embeddings**:
   - Run `app.py` to extract and save feature embeddings for all images in the dataset.
   ```bash
   python app.py
   ```

## Usage

### Web Application

1. **Run the Streamlit App**:
   ```bash
   streamlit run main.py
   ```

2. **Upload an Image**:
   - Use the Streamlit interface to upload an image, and the app will display five visually similar images from the dataset.

### Command-line Test Script

1. **Run the Test Script**:
   - The `test.py` script uses OpenCV to load a sample image, find similar images, and display them in a pop-up window.
   ```bash
   python test.py
   ```

2. **Input Image**:
   - Place a test image (e.g., `shirt.jpg`) in a folder named `sample` and adjust the file path in `test.py` if necessary.

## File Descriptions

- **app.py**: Generates and saves feature embeddings (`embeddings.pkl`) for images in the `images/` directory.
- **main.py**: Streamlit-based web application for uploading images and displaying recommendations.
- **test.py**: Local test script that performs similarity search on a sample image and displays results using OpenCV.
- **embeddings.pkl**: Contains precomputed feature vectors for images.
- **filenames.pkl**: Stores file paths of images corresponding to feature vectors in `embeddings.pkl`.

## Dependencies

- Python 3.6+
- TensorFlow
- Streamlit
- OpenCV
- Scikit-learn
- NumPy
- tqdm
- PIL

To install dependencies, use:
```bash
pip install -r requirements.txt
```

## Acknowledgments

This project utilizes:
- **ResNet50** from TensorFlow for feature extraction.
- **k-Nearest Neighbors** from Scikit-learn for similarity search.
- **Streamlit** for building an interactive web interface.
  
---
