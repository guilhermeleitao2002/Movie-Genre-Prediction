#!/usr/bin/env python3.11

import argparse
import re
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Clean the text
def clean_text(text):
    # Check if the text is a non-empty string
    if not isinstance(text, str) or not text.strip():
        return ''
    
    # Skip applying contractions if the text is too long or complex
    if len(text) > 500:  # Threshold to skip contraction expansion for long texts
        return text
    
    # Try expanding contractions safely
    try:
        text = contractions.fix(text)
    except Exception as e:
        print(f"Error expanding contractions: {e}, for text: {text[:100]}...")  # Only print the first 100 characters
        return text  # Return the original text if expansion fails
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

# Function to convert nltk POS tag to wordnet POS tag
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ  # Adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif tag.startswith('R'):
        return wordnet.ADV  # Adverb
    else:
        return wordnet.NOUN  # Default to noun

# Lemmatize the text
stop_words = set(stopwords.words('english'))

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Clean the text
    text = clean_text(text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    # Remove stop words and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Perform POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    # Lemmatize each token using the POS tag
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]
    return ' '.join(lemmatized_tokens)

# Load dataset
def load_data(filepath):
    data = read_csv(filepath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])
    return data

# Preprocess the data
def preprocess_data(data, lemmatize, combine_fields):
    # Extract relevant columns (plot, combine_fields, genre)
    selected_fields = ['plot'] + combine_fields
    if 'genre' in data.columns:
        selected_fields.append('genre')
    data = data[selected_fields].copy()
    
    # Handle missing values: drop rows with missing 'plot', fill missing combine_fields with ''
    data.dropna(subset=['plot'], inplace=True)  # Ensure 'plot' is not NaN
    for field in combine_fields:
        data[field] = data[field].fillna('')  # Replace NaN in combine_fields with empty strings

    # Combine plot and other specified fields into a single feature
    data['combined_text'] = data['plot']
    for field in combine_fields:
        data['combined_text'] += ' ' + data[field]

    # Apply lemmatization if specified
    if lemmatize:
        data['combined_text'] = data['combined_text'].apply(lemmatize_text)
    else:
        # Clean the text anyway
        data['combined_text'] = data['combined_text'].apply(clean_text)

    return data

# Split data into training and test sets
def split_data(data):
    X = data['combined_text']
    y = data['genre']
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded)
    return X_train, X_test, y_train, y_test, label_encoder

# Feature extraction using TF-IDF Vectorizer
def extract_features(X_train, X_test, max_features, ngram_range):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm='l2',
        min_df=5
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train SVM with hyperparameter tuning
def train_svm(X_train_tfidf, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear'],
        'class_weight': ['balanced', None]
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best SVM Parameters: {grid_search.best_params_}")
    return best_model

# Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm, report

# Save the best model
def save_model(model, vectorizer, label_encoder, model_path, vectorizer_path, label_encoder_path):
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    dump(label_encoder, label_encoder_path)

# Load the model and vectorizer
def load_model(model_path, vectorizer_path, label_encoder_path):
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    label_encoder = load(label_encoder_path)
    return model, vectorizer, label_encoder

# Predict the genre for new movie plots
def predict_genre(model, vectorizer, label_encoder, input_file, output_file, lemmatize, combine_fields):
    data = read_csv(input_file, sep='\t', names=['title', 'from', 'director', 'plot'])
    data = preprocess_data(data, lemmatize, combine_fields)
    plots = data['combined_text']
    plot_vectors = vectorizer.transform(plots)
    predicted_genres_encoded = model.predict(plot_vectors)
    predicted_genres = label_encoder.inverse_transform(predicted_genres_encoded)

    # Save the results to a file
    data['genre'] = predicted_genres
    data[['title', 'from', 'director', 'plot', 'genre']].to_csv(output_file, sep='\t', index=False)

# Main function to run the program
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a model for genre prediction based on movie plots.")
    parser.add_argument('--train_filepath', type=str, default='train.txt', help='Filepath for the training data.')
    parser.add_argument('--test_filepath', type=str, default='test_no_labels.txt', help='Filepath for the test data (without labels).')
    parser.add_argument('--results_filepath', type=str, default='results.txt', help='Filepath to save the results.')
    parser.add_argument('--max_features', '-f', type=int, default=10000, help='Maximum number of features for the TF-IDF vectorizer.')
    parser.add_argument('--ngram_range', '-n', type=str, default='1,5', help='N-gram range for the TF-IDF vectorizer, provided as "min_n,max_n".')
    parser.add_argument('--lemma', '-l', action='store_true', default=True, help='Whether to lemmatize the text data.')
    parser.add_argument('--combine_fields', '-c', type=str, default='from,director,title', help='Comma-separated fields to combine with the plot (e.g., "from,director,title").')
    parser.add_argument('--stop_words', '-s', type=str, default='english', help='Stop words for the TF-IDF vectorizer (e.g., "english" or "the,is,and").')
    args = parser.parse_args()

    # Parse ngram_range as a tuple
    ngram_range = tuple(map(int, args.ngram_range.split(',')))

    # Parse combine_fields as a list of field names
    combine_fields = args.combine_fields.split(',')

    # Convert the stop words string into a list (if not using 'english')
    if args.stop_words != 'english':
        stop_words = args.stop_words.split(',')
    else:
        stop_words = 'english'

    # Load and preprocess the data
    data = load_data(args.train_filepath)
    data = preprocess_data(data, args.lemma, combine_fields)

    # Analyze class distribution
    print("Class distribution:")
    print(data['genre'].value_counts())

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, label_encoder = split_data(data)

    # Extract features using TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test, args.max_features, ngram_range)

    # Train SVM with hyperparameter tuning
    best_svm_model = train_svm(X_train_tfidf, y_train)

    # Evaluate model
    svm_accuracy, svm_cm, svm_report = evaluate_model(best_svm_model, X_test_tfidf, y_test)
    print(f"Optimized SVM Accuracy: {svm_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(svm_report)

    # Cross-validation
    cv_scores = cross_val_score(best_svm_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Save the best model
    save_model(best_svm_model, vectorizer, label_encoder, 'best_model.pkl', 'vectorizer.pkl', 'label_encoder.pkl')

    # Predict genres for the test file and save results
    predict_genre(best_svm_model, vectorizer, label_encoder, args.test_filepath, args.results_filepath, args.lemma, combine_fields)
