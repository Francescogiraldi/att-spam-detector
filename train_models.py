import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import re
import string

# PyTorch and Transformers imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch or Transformers not available. Only scikit-learn model will be trained.")

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
    
    return model, tfidf

class SpamDataset(Dataset):
    """Dataset class for PyTorch"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_distilbert_model(X_train, X_test, y_train, y_test):
    """Train DistilBERT model"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping DistilBERT training.")
        return None, None
    
    print("\nTraining DistilBERT model...")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Create datasets
    train_dataset = SpamDataset(X_train.values, y_train.values, tokenizer)
    test_dataset = SpamDataset(X_test.values, y_test.values, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/3, Average Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    accuracy = correct / total
    print(f"DistilBERT model accuracy: {accuracy:.4f}")
    
    return model, tokenizer

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
    
    # Save scikit-learn model
    with open('models/sklearn_model.pkl', 'wb') as f:
        pickle.dump(sklearn_model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    print("Scikit-learn model saved!")
    
    # Train DistilBERT model if PyTorch is available
    if PYTORCH_AVAILABLE:
        distilbert_model, distilbert_tokenizer = train_distilbert_model(X_train, X_test, y_train, y_test)
        
        if distilbert_model is not None:
            # Save DistilBERT model
            distilbert_model.save_pretrained('models/distilbert_model')
            distilbert_tokenizer.save_pretrained('models/distilbert_tokenizer')
            print("DistilBERT model saved!")
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()