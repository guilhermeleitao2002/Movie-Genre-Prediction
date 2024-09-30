#!/usr/bin/env python3.11
from argparse import ArgumentParser
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
from nltk.stem import WordNetLemmatizer
#import nltk
#nltk.download('wordnet')


# Lemmatize the text
def lemmatize_text(text):
    # Lemmatizer function
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Load dataset
def load_data(filepath):
    data = read_csv(filepath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])
    return data

# Preprocess the data
def preprocess_data(data, lemmatize, combine_fields):
    # Extract relevant columns (title, plot, genre)
    selected_fields = ['plot'] + combine_fields
    selected_fields.append('genre')
    data = data[selected_fields].copy()
    data.dropna(inplace=True)

    # Combine plot and other specified fields into a single feature
    data['combined_text'] = data['plot']
    for field in combine_fields:
        data['combined_text'] += ' ' + data[field]

    # Apply lemmatization if specified
    if lemmatize:
        data['combined_text'] = data['combined_text'].apply(lemmatize_text)

    return data

# Split data into training and test sets
def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['genre'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Feature extraction using TF-IDF Vectorizer
def extract_features(X_train, X_test, max_features, ngram_range):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range)
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
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)

# Load the model and vectorizer
def load_model(model_path, vectorizer_path):
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    return model, vectorizer

# Predict the genre for new movie plots
def predict_genre(model, vectorizer, input_file, output_file):
    data = read_csv(input_file, sep='\t', names=['title', 'from', 'director', 'plot'])
    plots = data['plot']
    plot_vectors = vectorizer.transform(plots)
    predicted_genres = model.predict(plot_vectors)
    
    # Save the results to a file
    data['genre'] = predicted_genres
    data[['title', 'from', 'director', 'plot', 'genre']].to_csv(output_file, sep='\t', index=False)


# Main function to run the program
if __name__ == '__main__':
    # Parse command-line arguments
    parser = ArgumentParser(description="Train and evaluate a model for genre prediction based on movie plots.")
    parser.add_argument('--train_filepath', type=str, default='train.txt', help='Filepath for the training data.')
    parser.add_argument('--test_filepath', type=str, default='test_no_labels.txt', help='Filepath for the test data (without labels).')
    parser.add_argument('--results_filepath', type=str, default='results.txt', help='Filepath to save the results.')
    parser.add_argument('--max_features', '-f', type=int, default=5000, help='Maximum number of features for the TF-IDF vectorizer.')
    parser.add_argument('--ngram_range', '-n', type=str, default='1,2', help='N-gram range for the TF-IDF vectorizer, provided as "min_n,max_n".')
    parser.add_argument('--lemma', '-l', type=bool, default=True, help='Whether to lemmatize the text data.')
    parser.add_argument('--combine_fields', '-c', type=str, default='from,director', help='Comma-separated fields to combine with the plot (e.g., "director" or "from,director,title").')
    parser.add_argument('--stop_words', '-s', type=str, default='english', help='Stop words for the TF-IDF vectorizer (e.g., "english" or "the,is,and").')

    args = parser.parse_args()

    # Parse ngram_range as a tuple
    ngram_range = tuple(map(int, args.ngram_range.split(',')))

    # Parse combine_fields as a list of field names
    combine_fields = args.combine_fields.split(',')

    # Parse stop_words as a list ONLY if a comma-separated string is provided
    stop_words = args.stop_words.split(',')
    if len(stop_words) == 1:
        stop_words = args.stop_words

    # Load and preprocess the data
    data = load_data(args.train_filepath)
    data = preprocess_data(data, args.lemma, combine_fields)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Extract features using TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test, args.max_features, ngram_range)
    
    # Train the models
    nb_model, svm_model = train_models(X_train_tfidf, y_train)
    
    # Evaluate both models
    nb_accuracy, nb_cm = evaluate_model(nb_model, X_test_tfidf, y_test)
    svm_accuracy, svm_cm = evaluate_model(svm_model, X_test_tfidf, y_test)
    
    print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
    
    best_model = nb_model if nb_accuracy > svm_accuracy else svm_model
    save_model(best_model, vectorizer, 'best_model.pkl', 'vectorizer.pkl')
    
    # Predict genres for the test file and save results
    predict_genre(best_model, vectorizer, args.test_filepath, args.results_filepath)
