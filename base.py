#!/usr/bin/env python3.11

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Lemmatizer function
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Use n-grams in TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))


# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])
    return data

# Preprocess the data
def preprocess_data(data):
    # Extract relevant columns (title, plot, genre)
    data = data[['title', 'director', 'plot', 'genre']].copy()
    data.dropna(inplace=True)

    # Combine plot and director into a single feature (text)
    data['plot_director'] =data['plot']+ ' ' + data['director']
    
    data['plot_director'] = data['plot_director'].apply(lemmatize_text)

    return data

# Split data into training and test sets
def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['plot_director'], data['genre'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Feature extraction using TF-IDF Vectorizer
def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train models (Naive Bayes and SVM)
def train_models(X_train_tfidf, y_train):
    # Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)
    
    return nb_model, svm_model

# Evaluate the models
def evaluate_model(model, X_test_tfidf, y_test):
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm

# Save the best model
def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Load the model and vectorizer
def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Predict the genre for new movie plots
def predict_genre(model, vectorizer, input_file, output_file):
    data = pd.read_csv(input_file, sep='\t', names=['title', 'from', 'director', 'plot'])
    plots = data['plot']
    plot_vectors = vectorizer.transform(plots)
    predicted_genres = model.predict(plot_vectors)
    
    # Save the results to a file
    data['genre'] = predicted_genres
    data[['title', 'from', 'director', 'plot', 'genre']].to_csv(output_file, sep='\t', index=False)

# Main function to run the program
if __name__ == '__main__':
    # Load and preprocess the data
    train_filepath = 'train.txt'
    test_filepath = 'test_no_labels.txt'
    results_filepath = 'results.txt'
    
    data = load_data(train_filepath)
    data = preprocess_data(data)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Extract features using TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    # Train the models
    nb_model, svm_model = train_models(X_train_tfidf, y_train)
    
    # Evaluate both models
    nb_accuracy, nb_cm = evaluate_model(nb_model, X_test_tfidf, y_test)
    svm_accuracy, svm_cm = evaluate_model(svm_model, X_test_tfidf, y_test)
    
    print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
    
    # Choose the better model (for this example, we assume SVM is better)
    best_model = svm_model
    save_model(best_model, vectorizer, 'best_model.pkl', 'vectorizer.pkl')
    
    # Predict genres for the test file and save results
    predict_genre(best_model, vectorizer, test_filepath, results_filepath)
