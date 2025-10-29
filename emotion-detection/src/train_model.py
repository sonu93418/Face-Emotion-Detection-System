"""
Emotion Detection Model Training Script
Implements TF-IDF vectorization and Logistic Regression for emotion classification
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import TextPreprocessor, create_sample_dataset

class EmotionClassifier:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the Emotion Classifier
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): Range of n-grams for TF-IDF
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english'
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def load_data(self, file_path=None, use_sample=True):
        """
        Load emotion dataset
        
        Args:
            file_path (str): Path to CSV file with emotion data
            use_sample (bool): Whether to use sample data if file not found
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path and os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        else:
            if use_sample:
                print("Using sample dataset for demonstration")
                df = create_sample_dataset()
                # Create a larger sample dataset by duplicating with variations
                df = self._expand_sample_dataset(df)
            else:
                raise FileNotFoundError("Dataset file not found and sample data disabled")
        
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
        
        return df
    
    def _expand_sample_dataset(self, df):
        """
        Expand the sample dataset for better training
        """
        # Additional sample data for better model training
        additional_data = {
            'text': [
                "I feel fantastic today! Everything is going perfectly!",
                "This is so annoying. Why does this always happen to me?",
                "I'm terrified of what might happen next.",
                "Feeling blue and down today. Nothing seems right.",
                "Absolutely delighted with the results!",
                "I'm furious about this situation!",
                "Just another ordinary day at work.",
                "Ecstatic about the good news!",
                "This makes me so mad I could scream!",
                "Anxious about the upcoming changes.",
                "Thrilled beyond words!",
                "Heartbroken about what happened.",
                "Enraged by the unfair treatment!",
                "Worried sick about the outcome.",
                "Pure bliss and happiness!",
                "Devastated by the recent events.",
                "Livid about the poor service!",
                "Panicked about the deadline.",
                "Neutral feelings about the proposal.",
                "Elated with the achievement!",
                "Melancholy mood today.",
                "Outraged by the behavior!",
                "Fearful of the consequences.",
                "Cheerful and optimistic!",
                "Gloomy atmosphere around here.",
                "Irate customer service experience!",
                "Apprehensive about the meeting.",
                "Jubilant celebration time!",
                "Somber reflection on events.",
                "Incensed by the decision!"
            ],
            'emotion': [
                'joy', 'anger', 'fear', 'sadness', 'joy', 'anger', 'neutral', 'joy',
                'anger', 'fear', 'joy', 'sadness', 'anger', 'fear', 'joy', 'sadness',
                'anger', 'fear', 'neutral', 'joy', 'sadness', 'anger', 'fear', 'joy',
                'sadness', 'anger', 'fear', 'joy', 'sadness', 'anger'
            ]
        }
        
        additional_df = pd.DataFrame(additional_data)
        combined_df = pd.concat([df, additional_df], ignore_index=True)
        
        return combined_df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Preprocessing data...")
        
        # Remove any missing values
        df = df.dropna(subset=['text', 'emotion'])
        
        # Preprocess text
        processed_df = self.preprocessor.preprocess_dataframe(df, 'text', 'emotion')
        
        return processed_df
    
    def prepare_features(self, df, fit_vectorizer=True):
        """
        Prepare features using TF-IDF vectorization
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            fit_vectorizer (bool): Whether to fit the vectorizer
            
        Returns:
            tuple: Features and labels
        """
        print("Preparing features with TF-IDF vectorization...")
        
        # Use cleaned text for vectorization
        texts = df['cleaned_text'].values
        
        if fit_vectorizer:
            X = self.vectorizer.fit_transform(texts)
            print(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        else:
            X = self.vectorizer.transform(texts)
        
        # Encode labels
        if fit_vectorizer:  # Only fit encoder during training
            y = self.label_encoder.fit_transform(df['emotion'].values)
            print(f"Emotion classes: {self.label_encoder.classes_}")
        else:
            y = self.label_encoder.transform(df['emotion'].values)
        
        print(f"Feature matrix shape: {X.shape}")
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """
        Train the Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training Logistic Regression model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Model training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Emotion Detection')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('../models', exist_ok=True)
        plt.savefig('../models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to ../models/confusion_matrix.png")
        plt.show()
    
    def predict_emotion(self, text):
        """
        Predict emotion for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (predicted_emotion, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Convert back to emotion label
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create confidence dictionary
        confidence_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            confidence_dict[class_name] = probabilities[i]
        
        return emotion, confidence_dict
    
    def save_model(self, model_dir='../models'):
        """
        Save the trained model and components
        
        Args:
            model_dir (str): Directory to save model files
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, os.path.join(model_dir, 'emotion_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='../models'):
        """
        Load a pre-trained model
        
        Args:
            model_dir (str): Directory containing model files
        """
        try:
            self.model = joblib.load(os.path.join(model_dir, 'emotion_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            self.is_trained = True
            print(f"Model loaded from {model_dir}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found in {model_dir}")

def main():
    """
    Main training pipeline
    """
    print("=== Emotion Detection Model Training ===\n")
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Load and preprocess data
    df = classifier.load_data(use_sample=True)
    processed_df = classifier.preprocess_data(df)
    
    # Prepare features
    X, y = classifier.prepare_features(processed_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    classifier.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    print("\n=== Testing Sample Predictions ===")
    test_texts = [
        "I'm absolutely thrilled about this opportunity!",
        "This situation makes me so angry and frustrated!",
        "I'm really scared about what might happen next.",
        "Feeling quite sad and depressed today.",
        "It's just an ordinary day, nothing special."
    ]
    
    for text in test_texts:
        emotion, confidence = classifier.predict_emotion(text)
        print(f"\nText: '{text}'")
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence[emotion]:.4f}")

if __name__ == "__main__":
    main()