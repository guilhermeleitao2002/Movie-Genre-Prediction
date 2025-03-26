# Movie Genre Classification Project

## Overview
This project aims to classify movie plots into one of nine genres: drama, comedy, horror, action, romance, western, animation, crime, or sci-fi. The goal is to build a Python-based model that predicts genres for movie plots provided in a test set. 

## Dataset
The training data (`train.txt`) contains the following tab-separated fields per line:
- **Title**: Movie title.
- **From**: Origin (e.g., "British drama").
- **Genre**: Target label (one of the nine genres).
- **Director**: Movie director.
- **Plot**: Full text of the movie plot.

The test set (`test_no_labels.txt`) contains plots without genre labels. The model must generate predictions for these plots and save them in `results.txt`, where each line corresponds to the predicted genre for the respective plot in the test file.

## Implementation Steps

### 1. Preprocessing
- Clean and normalize text data (e.g., lowercasing, removing special characters).
- Handle potential inconsistencies in the dataset (e.g., unbalanced classes, mislabeled entries).
- Optional: Experiment with techniques like stop-word removal, lemmatization, or TF-IDF vectorization.

### 2. Model Development
- Use Python 3 to build a classifier. Suggested approaches include:
  - Traditional ML models (e.g., SVM, Naive Bayes, Logistic Regression) with TF-IDF or CountVectorizer.
  - Deep learning models (e.g., LSTM, Transformer-based architectures) for text classification.
- Split the training data into custom train/dev/test sets for iterative evaluation.
- Tune hyperparameters and experiment with feature engineering to improve performance.

### 3. Prediction Generation
- Apply the trained model to `test_no_labels.txt`.
- Ensure predictions in `results.txt` follow the same line order as the input test file.

## Code Structure
- **Main Script**: `reviews.py` (reads input, runs the model, and writes predictions to `results.txt`).
- **Additional Files**: Optional helper modules (e.g., `preprocess.py`, `model.py`).

## Dependencies
- Python 3.x
- Libraries such as `scikit-learn`, `nltk`, `pandas`, `tensorflow`, or `transformers` (depending on the chosen approach).

## Usage
1. Place the training data (`train.txt`) and test data (`test_no_labels.txt`) in the project directory.
2. Run the main script:
   ```bash
   python reviews.py
   ```
3. The predictions will be saved to `results.txt`.

## Notes
- Ensure preprocessing steps applied to the training data are replicated on the test set.
- Experiment with multiple models and document findings in the accompanying paper (not part of this README).
