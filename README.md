# CNN on CIFAR-10

This repository contains my submission for **Stage 2A of the AI Studio Selection Process**.  
The objective was to train a small CNN (or comparable baseline) on a subset of the CIFAR-10 dataset and document the experimentation process, observations, and reasoning rather than optimizing purely for accuracy.

##  Problem Overview

CIFAR-10 is a standard image classification dataset consisting of 60,000 color images (32×32 pixels) across 10 object categories such as airplanes, automobiles, animals, and ships.

The task was approached as a **learning-oriented experiment**, focusing on:
- establishing a baseline model,
- identifying its limitations,
- improving performance through architectural changes,
- and reflecting on observed trade-offs.

##  Approach

### 1. Baseline Model — Artificial Neural Network (ANN)
A fully connected ANN was trained as an initial baseline to understand how a simple model performs on image data.

**Observation:**  
While the ANN learned basic patterns, its performance was limited because it treats each pixel independently and does not capture spatial relationships inherent in images.

### 2. Convolutional Neural Network (CNN) — Primary Model
A CNN with convolutional and pooling layers was then implemented to leverage spatial feature extraction.

**Result:**  
The CNN significantly outperformed the ANN, achieving ~68% test accuracy by learning hierarchical features such as edges and textures.

### 3. Experiment — CNN with Dropout
To study generalization, Dropout (0.5) was added to the dense layer.

**Observation:**  
Test accuracy decreased slightly (~66.3%), but training and validation curves became more stable, indicating reduced overfitting. This highlighted the trade-off between model capacity and robustness, especially under limited training epochs.

##  Training Curves

The notebook includes:
- Training vs validation accuracy plots
- Training vs validation loss plots  
for both the CNN without dropout (primary model) and CNN with dropout (experiment).

These curves were used to reason about overfitting, stability, and generalization rather than to optimize final accuracy.

## Key Learnings

- Baseline models are essential for understanding task complexity.
- CNNs are better suited for image data due to spatial feature extraction.
- Regularization techniques like Dropout may reduce raw accuracy but improve robustness.
- Model evaluation should consider training dynamics and generalization, not accuracy alone.

## Tools & Frameworks

- Python
- TensorFlow / Keras
- Google Colab
- CIFAR-10 Dataset

## Files

- `Stage2A_CNN_CIFAR10.ipynb` — Colab notebook containing experiments, plots, and observations
- `README.md` — Project overview and reasoning

## Notes

This project prioritizes **learning, experimentation, and reasoning** over leaderboard-style optimization.  
Failed or suboptimal experiments are intentionally documented as part of the learning process.


# CNN Grad-CAM Interpretability Demo

A Streamlit-based web application that visualizes Convolutional Neural Network (CNN) decision-making using Grad-CAM. This project helps demystify "black-box" models by generating heatmaps that highlight the image regions most influential to the model's classification.

## Features
- **Interactive Interface**: Upload custom images for real-time classification.
- **Grad-CAM Visualization**: Generates heatmaps highlighting feature importance.
- **Explainability**: Provides technical insights into how the model perceives the input data.

## Project Structure
- `app.py`: The Streamlit web application UI.
- `model_utils.py`: Contains the core Grad-CAM logic.
- `saved_model.keras`: The serialized Keras model.
- `requirements.txt`: Project dependencies.

## Installation
1. Clone the repo: `git clone https://github.com/vani-max/CNN_Model_Project.git`
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`
