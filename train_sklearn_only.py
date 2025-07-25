import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re
import string

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data():
    """Load and preprocess the SMS spam dataset"""
    print("Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Keep only the first two columns and rename them
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean the text
    df['message'] = df['message'].apply(clean_text)
    
    print(f"Dataset loaded: {len(df)} messages")
    print(f"Spam messages: {df['label'].sum()}")
    print(f"Ham messages: {len(df) - df['label'].sum()}")
    
    return df

def train_sklearn_model(X_train, X_test, y_train, y_test):
    """Train scikit-learn model with TF-IDF"""
    print("\nTraining scikit-learn model...")
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Scikit-learn model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    return model, tfidf

def main():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split the data
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train scikit-learn model
    sklearn_model, tfidf_vectorizer = train_sklearn_model(X_train, X_test, y_train, y_test)
    
    # Save scikit-learn model using joblib
    joblib.dump(sklearn_model, 'models/sklearn_model.joblib')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
    
    print("\nScikit-learn model saved successfully!")
    print("Models saved in 'models/' directory:")
    print("- sklearn_model.joblib")
    print("- tfidf_vectorizer.joblib")

if __name__ == "__main__":
    main()