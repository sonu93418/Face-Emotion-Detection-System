"""
Model Evaluation Script for Emotion Detection
Comprehensive evaluation with accuracy, precision, recall, F1-score, and confusion matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os

from train_model import EmotionClassifier
from preprocessing import create_sample_dataset

class ModelEvaluator:
    def __init__(self, model_path='../models'):
        """
        Initialize the model evaluator
        
        Args:
            model_path (str): Path to saved model files
        """
        self.model_path = model_path
        self.classifier = None
        self.evaluation_results = {}
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.classifier = EmotionClassifier()
            self.classifier.load_model(self.model_path)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model not found. Training a new model...")
            self.train_and_save_model()
    
    def train_and_save_model(self):
        """Train and save a new model if none exists"""
        self.classifier = EmotionClassifier()
        
        # Load sample data
        df = self.classifier.load_data(use_sample=True)
        processed_df = self.classifier.preprocess_data(df)
        X, y = self.classifier.prepare_features(processed_df)
        
        # Train the model
        self.classifier.train_model(X, y)
        
        # Save the model
        self.classifier.save_model(self.model_path)
        print("New model trained and saved!")
    
    def evaluate_comprehensive(self, test_data=None):
        """
        Perform comprehensive model evaluation
        
        Args:
            test_data (pd.DataFrame): Test dataset (optional)
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if self.classifier is None:
            self.load_model()
        
        # Use sample data if no test data provided
        if test_data is None:
            test_data = create_sample_dataset()
        
        # Preprocess test data
        processed_data = self.classifier.preprocessor.preprocess_dataframe(
            test_data, 'text', 'emotion'
        )
        
        # Prepare features
        X_test, y_test = self.classifier.prepare_features(processed_data, fit_vectorizer=False)
        
        # Make predictions
        y_pred = self.classifier.model.predict(X_test)
        y_pred_proba = self.classifier.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.classifier.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test,
            'probabilities': y_pred_proba
        }
        
        return self.evaluation_results
    
    def cross_validation_evaluation(self, cv_folds=5):
        """
        Perform cross-validation evaluation
        
        Args:
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        if self.classifier is None:
            self.load_model()
        
        # Load data
        df = create_sample_dataset()
        processed_df = self.classifier.preprocessor.preprocess_dataframe(df, 'text', 'emotion')
        X, y = self.classifier.prepare_features(processed_df)
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.classifier.model, X, y, 
            cv=cv, scoring='accuracy'
        )
        
        cv_precision = cross_val_score(
            self.classifier.model, X, y,
            cv=cv, scoring='precision_macro'
        )
        
        cv_recall = cross_val_score(
            self.classifier.model, X, y,
            cv=cv, scoring='recall_macro'
        )
        
        cv_f1 = cross_val_score(
            self.classifier.model, X, y,
            cv=cv, scoring='f1_macro'
        )
        
        cv_results = {
            'accuracy_scores': cv_scores,
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'precision_scores': cv_precision,
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_scores': cv_recall,
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'f1_scores': cv_f1,
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
        
        return cv_results
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot and save confusion matrix
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Run evaluate_comprehensive first.")
        
        cm = self.evaluation_results['confusion_matrix']
        classes = self.classifier.label_encoder.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Emotion Detection Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_metrics(self, save_path=None):
        """
        Plot classification metrics (precision, recall, F1-score) by class
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Run evaluate_comprehensive first.")
        
        report = self.evaluation_results['classification_report']
        classes = self.classifier.label_encoder.classes_
        
        # Extract metrics for each class
        precision_scores = [report[cls]['precision'] for cls in classes]
        recall_scores = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)        # Instead of 50 samples, you need 1000s like:
        emotions = {
            'joy': 5000,
            'anger': 5000, 
            'fear': 5000,
            'sadness': 5000,
            'neutral': 5000
        }
        ax.set_title('Classification Metrics by Emotion Class', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification metrics plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, save_path=None):
        """
        Generate a comprehensive evaluation report
        
        Args:
            save_path (str): Path to save the report
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Run evaluate_comprehensive first.")
        
        results = self.evaluation_results
        
        # Generate report
        report = f"""
# Emotion Detection Model - Evaluation Report

## Overall Performance Metrics

- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **Precision (Macro)**: {results['precision_macro']:.4f}
- **Precision (Weighted)**: {results['precision_weighted']:.4f}
- **Recall (Macro)**: {results['recall_macro']:.4f}
- **Recall (Weighted)**: {results['recall_weighted']:.4f}
- **F1-Score (Macro)**: {results['f1_macro']:.4f}
- **F1-Score (Weighted)**: {results['f1_weighted']:.4f}

## Classification Report by Emotion Class

"""
        
        # Add per-class metrics
        class_report = results['classification_report']
        classes = self.classifier.label_encoder.classes_
        
        for cls in classes:
            if cls in class_report:
                report += f"""
### {cls.title()} Emotion
- Precision: {class_report[cls]['precision']:.4f}
- Recall: {class_report[cls]['recall']:.4f}
- F1-Score: {class_report[cls]['f1-score']:.4f}
- Support: {class_report[cls]['support']} samples
"""
        
        # Add confusion matrix analysis
        cm = results['confusion_matrix']
        report += f"""

## Confusion Matrix Analysis

Total Test Samples: {cm.sum()}
Correctly Classified: {np.diag(cm).sum()}
Misclassified: {cm.sum() - np.diag(cm).sum()}

"""
        
        # Model strengths and weaknesses
        report += """
## Model Analysis

### Strengths:
- Good overall accuracy for emotion detection
- Balanced performance across multiple emotion classes
- Effective text preprocessing and feature extraction

### Areas for Improvement:
- Consider using more training data for better generalization
- Experiment with advanced models like neural networks
- Fine-tune hyperparameters for optimal performance

### Recommendations:
1. Collect more diverse training data
2. Try ensemble methods for improved accuracy
3. Implement real-time model updating capabilities
4. Add more sophisticated feature engineering techniques
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Evaluation report saved to {save_path}")
        
        return report
    
    def test_sample_predictions(self):
        """Test the model with sample predictions"""
        if self.classifier is None:
            self.load_model()
        
        test_texts = [
            "I'm absolutely thrilled and excited about this amazing opportunity!",
            "This situation is making me extremely frustrated and angry!",
            "I'm really worried and scared about what might happen next.",
            "Feeling quite sad and depressed today, nothing seems right.",
            "It's just another ordinary day, nothing particularly exciting.",
            "I love spending time with my family during the holidays!",
            "This unfair treatment is making me furious and upset!",
            "The thought of that surgery next week terrifies me completely.",
            "I'm heartbroken about losing my best friend recently.",
            "The weather today is okay, not too hot or cold."
        ]
        
        expected_emotions = [
            'joy', 'anger', 'fear', 'sadness', 'neutral',
            'joy', 'anger', 'fear', 'sadness', 'neutral'
        ]
        
        print("\n=== Sample Prediction Tests ===\n")
        
        correct_predictions = 0
        total_predictions = len(test_texts)
        
        for i, (text, expected) in enumerate(zip(test_texts, expected_emotions)):
            predicted_emotion, confidence = self.classifier.predict_emotion(text)
            is_correct = predicted_emotion.lower() == expected.lower()
            
            if is_correct:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"Test {i+1}: {status}")
            print(f"Text: '{text[:60]}...'")
            print(f"Expected: {expected} | Predicted: {predicted_emotion}")
            print(f"Confidence: {confidence[predicted_emotion]:.3f}")
            print("-" * 60)
        
        sample_accuracy = correct_predictions / total_predictions
        print(f"\nSample Test Accuracy: {sample_accuracy:.2%} ({correct_predictions}/{total_predictions})")
        
        return sample_accuracy

def main():
    """
    Main evaluation pipeline
    """
    print("=== Emotion Detection Model Evaluation ===\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    evaluator.load_model()
    
    # Comprehensive evaluation
    print("1. Running comprehensive evaluation...")
    results = evaluator.evaluate_comprehensive()
    
    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"Precision (Macro): {results['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    
    # Cross-validation
    print("\n2. Running cross-validation...")
    cv_results = evaluator.cross_validation_evaluation()
    print(f"CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"CV F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    os.makedirs('../models', exist_ok=True)
    
    evaluator.plot_confusion_matrix('../models/confusion_matrix_detailed.png')
    evaluator.plot_classification_metrics('../models/classification_metrics.png')
    
    # Generate report
    print("\n4. Generating evaluation report...")
    report = evaluator.generate_evaluation_report('../models/evaluation_report.md')
    
    # Test sample predictions
    print("\n5. Testing sample predictions...")
    sample_accuracy = evaluator.test_sample_predictions()
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"📈 Overall Model Performance: {results['accuracy']*100:.1f}%")
    print(f"📝 Report saved to: ../models/evaluation_report.md")
    print(f"📊 Visualizations saved to: ../models/")

if __name__ == "__main__":
    main()