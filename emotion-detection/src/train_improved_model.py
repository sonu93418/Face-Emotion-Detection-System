"""
Improved Training Script with Enhanced Dataset and Better Model Performance
Uses larger, balanced dataset for significantly better accuracy
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import TextPreprocessor
from enhanced_dataset import create_enhanced_emotion_dataset, create_mega_dataset

class ImprovedEmotionClassifier:
    def __init__(self, max_features=10000, ngram_range=(1, 3), model_type='logistic'):
        """
        Initialize the Improved Emotion Classifier
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): Range of n-grams for TF-IDF
            model_type (str): Type of model ('logistic', 'random_forest', 'svm')
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Choose model based on type
        if model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=2000,
                solver='lbfgs',
                multi_class='multinomial'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                C=1.0
            )
        else:
            raise ValueError("model_type must be 'logistic', 'random_forest', or 'svm'")
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_type = model_type
        
    def load_enhanced_data(self, dataset_type='mega'):
        """
        Load enhanced emotion dataset
        
        Args:
            dataset_type (str): 'enhanced' for 150 samples or 'mega' for 200+ samples
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading {dataset_type} emotion dataset...")
        
        if dataset_type == 'mega':
            df = create_mega_dataset()
        else:
            df = create_enhanced_emotion_dataset()
        
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset with improved cleaning
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Preprocessing data with enhanced pipeline...")
        
        # Remove any missing values
        df = df.dropna(subset=['text', 'emotion'])
        
        # Preprocess text
        processed_df = self.preprocessor.preprocess_dataframe(df, 'text', 'emotion')
        
        # Remove very short texts (less than 2 words after preprocessing)
        processed_df = processed_df[processed_df['cleaned_text'].str.split().str.len() >= 2]
        
        print(f"After preprocessing: {len(processed_df)} samples remaining")
        
        return processed_df
    
    def prepare_features_advanced(self, df, fit_vectorizer=True):
        """
        Prepare features using advanced TF-IDF vectorization
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            fit_vectorizer (bool): Whether to fit the vectorizer
            
        Returns:
            tuple: Features and labels
        """
        print("Preparing features with advanced TF-IDF vectorization...")
        
        # Use cleaned text for vectorization
        texts = df['cleaned_text'].values
        
        if fit_vectorizer:
            X = self.vectorizer.fit_transform(texts)
            print(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
            print(f"Feature matrix shape: {X.shape}")
            
            # Show top features for each emotion
            self._analyze_top_features(df)
        else:
            X = self.vectorizer.transform(texts)
        
        # Encode labels
        if fit_vectorizer:  # Only fit encoder during training
            y = self.label_encoder.fit_transform(df['emotion'].values)
            print(f"Emotion classes: {self.label_encoder.classes_}")
        else:
            y = self.label_encoder.transform(df['emotion'].values)
        
        return X, y
    
    def _analyze_top_features(self, df):
        """
        Analyze top TF-IDF features for each emotion
        """
        print("\n📊 Top TF-IDF features by emotion:")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        for emotion in df['emotion'].unique():
            emotion_texts = df[df['emotion'] == emotion]['cleaned_text']
            emotion_tfidf = self.vectorizer.transform(emotion_texts)
            
            # Get mean TF-IDF scores for this emotion
            mean_scores = np.mean(emotion_tfidf.toarray(), axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            print(f"  {emotion.title()}: {', '.join(top_features[:5])}")
    
    def train_with_hyperparameter_tuning(self, X_train, y_train):
        """
        Train model with hyperparameter tuning for better performance
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training {self.model_type} model with hyperparameter tuning...")
        
        if self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Use best model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print("Model training completed!")
    
    def evaluate_model_comprehensive(self, X_test, y_test):
        """
        Comprehensive model evaluation with detailed metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model performance comprehensively...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n🎯 Model Performance Results:")
        print(f"Model Type: {self.model_type.title()}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        print(f"\n📊 Per-Class Performance:")
        for emotion in self.label_encoder.classes_:
            if emotion in report:
                precision = report[emotion]['precision']
                recall = report[emotion]['recall']
                f1 = report[emotion]['f1-score']
                support = report[emotion]['support']
                
                print(f"  {emotion.title():>8}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={support})")
        
        # Plot confusion matrix
        self._plot_enhanced_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def _plot_enhanced_confusion_matrix(self, cm):
        """
        Plot an enhanced confusion matrix with better styling
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
            annotations.append(row)
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Enhanced Confusion Matrix - {self.model_type.title()} Model', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('../models', exist_ok=True)
        filename = f'../models/enhanced_confusion_matrix_{self.model_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Enhanced confusion matrix saved to {filename}")
        plt.show()
    
    def predict_emotion_with_confidence(self, text):
        """
        Predict emotion with detailed confidence analysis
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Detailed prediction results
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
        
        # Create detailed confidence dictionary
        confidence_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            confidence_dict[class_name] = probabilities[i]
        
        # Sort by confidence
        sorted_confidence = dict(sorted(confidence_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return {
            'predicted_emotion': emotion,
            'confidence_score': confidence_dict[emotion],
            'all_confidences': sorted_confidence,
            'cleaned_text': cleaned_text,
            'original_text': text
        }
    
    def save_improved_model(self, model_dir='../models'):
        """
        Save the improved trained model
        
        Args:
            model_dir (str): Directory to save model files
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components with model type suffix
        model_suffix = f'_{self.model_type}'
        joblib.dump(self.model, os.path.join(model_dir, f'improved_emotion_model{model_suffix}.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, f'improved_tfidf_vectorizer{model_suffix}.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, f'improved_label_encoder{model_suffix}.pkl'))
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'max_features': self.vectorizer.max_features,
            'ngram_range': self.vectorizer.ngram_range,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'classes': self.label_encoder.classes_.tolist()
        }
        
        import json
        with open(os.path.join(model_dir, f'model_metadata{model_suffix}.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Improved {self.model_type} model saved to {model_dir}")

def main():
    """
    Main improved training pipeline
    """
    print("=== IMPROVED Emotion Detection Model Training ===\n")
    
    # Test different model types
    model_types = ['logistic', 'random_forest']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        # Initialize improved classifier
        classifier = ImprovedEmotionClassifier(
            max_features=10000,
            ngram_range=(1, 3),
            model_type=model_type
        )
        
        # Load enhanced dataset
        df = classifier.load_enhanced_data('mega')
        
        # Preprocess data
        processed_df = classifier.preprocess_data(df)
        
        # Prepare features
        X, y = classifier.prepare_features_advanced(processed_df)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train model with hyperparameter tuning
        classifier.train_with_hyperparameter_tuning(X_train, y_train)
        
        # Evaluate model
        metrics = classifier.evaluate_model_comprehensive(X_test, y_test)
        results[model_type] = metrics['accuracy']
        
        # Save model
        classifier.save_improved_model()
        
        # Test sample predictions
        print(f"\n=== Testing Sample Predictions ({model_type}) ===")
        test_texts = [
            "I'm absolutely thrilled about this amazing opportunity!",
            "This situation makes me so angry and frustrated!",
            "I'm really scared about what might happen next.",
            "Feeling quite sad and depressed today.",
            "It's just an ordinary day, nothing special happening."
        ]
        
        for text in test_texts:
            result = classifier.predict_emotion_with_confidence(text)
            print(f"\nText: '{text}'")
            print(f"Predicted: {result['predicted_emotion']} (confidence: {result['confidence_score']:.3f})")
            
            # Show top 3 predictions
            top_3 = list(result['all_confidences'].items())[:3]
            print(f"Top 3: {', '.join([f'{e}({c:.2f})' for e, c in top_3])}")
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for model_type, accuracy in results.items():
        print(f"{model_type.title()} Model: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model.title()} with {results[best_model]*100:.1f}% accuracy")

if __name__ == "__main__":
    main()