"""
Utility functions and helper scripts for the Emotion Detection project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

def create_extended_sample_dataset():
    """
    Create a more comprehensive sample dataset for better model training
    """
    # Extended sample data with more diverse examples
    extended_data = {
        'text': [
            # Joy examples
            "I'm absolutely thrilled about this amazing opportunity!",
            "Feeling fantastic today! Everything is going perfectly!",
            "So excited about the weekend plans with friends!",
            "Just got promoted at work! Best day ever!",
            "My baby took her first steps today! Pure joy!",
            "Won the lottery! Can't believe this is happening!",
            "Amazing vacation in Hawaii! Living the dream!",
            "Just married the love of my life! Ecstatic!",
            "Graduated with honors! All the hard work paid off!",
            "Surprise party was incredible! Feeling blessed!",
            
            # Anger examples
            "This situation is making me extremely frustrated!",
            "Can't believe how unfair this treatment is!",
            "Traffic jam is making me furious! Running late!",
            "Terrible customer service! Completely unacceptable!",
            "My computer crashed and I lost all my work!",
            "Neighbors are being incredibly inconsiderate!",
            "Got charged twice for the same purchase! Outrageous!",
            "Boss rejected my proposal without even reading it!",
            "Parking ticket for no good reason! This is ridiculous!",
            "Internet has been down for hours! So frustrating!",
            
            # Fear examples
            "Really scared about the upcoming surgery next week.",
            "Terrified of flying but have to take this business trip.",
            "Worried sick about my job interview tomorrow.",
            "The storm outside is making me anxious and fearful.",
            "Afraid I won't pass the final exam despite studying.",
            "Nervous about moving to a new city all alone.",
            "Scared of what the test results might reveal.",
            "Anxious about giving a presentation to the board.",
            "Frightened by the strange noises in the house.",
            "Panic attack thinking about the deadline approaching.",
            
            # Sadness examples
            "Feeling really down and depressed today.",
            "Heartbroken about losing my beloved pet.",
            "So sad to see my best friend moving away.",
            "Mourning the loss of my grandmother last week.",
            "Disappointed that my vacation got cancelled.",
            "Feeling lonely since my roommate moved out.",
            "Devastated by the news about the company closing.",
            "Melancholy mood because of the rainy weather.",
            "Sad to see how divided our community has become.",
            "Gloomy atmosphere after hearing the bad news.",
            
            # Neutral examples
            "Just another ordinary day at the office.",
            "Weather is okay today, not too hot or cold.",
            "Had lunch, now back to work on the project.",
            "Commute was normal, no delays or issues.",
            "Finished reading a book, it was alright.",
            "Went to the store, bought some groceries.",
            "Attended the meeting, discussed quarterly results.",
            "Watched a documentary about history last night.",
            "Completed the daily tasks as scheduled.",
            "Regular workout session at the gym today."
        ],
        'emotion': [
            # Joy labels
            'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy',
            # Anger labels  
            'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger',
            # Fear labels
            'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
            # Sadness labels
            'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness',
            # Neutral labels
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
        ]
    }
    
    return pd.DataFrame(extended_data)

def analyze_text_statistics(df, text_column='text'):
    """
    Analyze basic statistics of the text data
    
    Args:
        df (pd.DataFrame): Dataset with text data
        text_column (str): Name of the text column
    
    Returns:
        dict: Text statistics
    """
    texts = df[text_column].astype(str)
    
    # Basic statistics
    word_counts = texts.str.split().str.len()
    char_counts = texts.str.len()
    
    stats = {
        'total_samples': len(df),
        'avg_words_per_text': word_counts.mean(),
        'avg_chars_per_text': char_counts.mean(),
        'min_words': word_counts.min(),
        'max_words': word_counts.max(),
        'min_chars': char_counts.min(),
        'max_chars': char_counts.max()
    }
    
    return stats

def plot_emotion_distribution(df, emotion_column='emotion', save_path=None):
    """
    Plot the distribution of emotions in the dataset
    
    Args:
        df (pd.DataFrame): Dataset with emotion labels
        emotion_column (str): Name of the emotion column
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    emotion_counts = df[emotion_column].value_counts()
    colors = ['#28a745', '#dc3545', '#6c757d', '#fd7e14', '#17a2b8']
    
    bars = plt.bar(emotion_counts.index, emotion_counts.values, 
                   color=colors[:len(emotion_counts)])
    
    plt.title('Distribution of Emotions in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Emotion Categories', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Emotion distribution plot saved to {save_path}")
    
    plt.show()

def create_word_cloud(df, emotion=None, text_column='text', save_path=None):
    """
    Create a word cloud for the dataset or specific emotion
    
    Args:
        df (pd.DataFrame): Dataset with text data
        emotion (str): Specific emotion to filter (optional)
        text_column (str): Name of the text column
        save_path (str): Path to save the word cloud
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("WordCloud not installed. Install with: pip install wordcloud")
        return
    
    # Filter by emotion if specified
    if emotion:
        filtered_df = df[df['emotion'].str.lower() == emotion.lower()]
        title = f"Word Cloud - {emotion.title()} Emotion"
    else:
        filtered_df = df
        title = "Word Cloud - All Emotions"
    
    if len(filtered_df) == 0:
        print(f"No data found for emotion: {emotion}")
        return
    
    # Combine all texts
    all_text = ' '.join(filtered_df[text_column].astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5
    ).generate(all_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
    plt.show()

def get_model_info():
    """
    Get information about the trained model
    
    Returns:
        dict: Model information
    """
    model_info = {
        'algorithm': 'Logistic Regression',
        'preprocessing': [
            'Text cleaning (URLs, mentions, punctuation removal)',
            'Tokenization',
            'Stopword removal', 
            'Lemmatization'
        ],
        'feature_extraction': 'TF-IDF Vectorization',
        'max_features': 5000,
        'ngram_range': '(1, 2)',
        'target_emotions': ['joy', 'anger', 'fear', 'sadness', 'neutral'],
        'expected_accuracy': '85-90%'
    }
    
    return model_info

def print_project_summary():
    """
    Print a comprehensive project summary
    """
    print("="*60)
    print("🎭 EMOTION DETECTION FROM TEXT - PROJECT SUMMARY")
    print("="*60)
    
    print("\n📋 PROJECT OVERVIEW:")
    print("• Detect emotions (joy, anger, fear, sadness, neutral) from text")
    print("• Uses Machine Learning with Natural Language Processing")
    print("• Interactive Streamlit web application")
    print("• Comprehensive evaluation with multiple metrics")
    
    print("\n🛠️ TECHNOLOGIES:")
    print("• Python 3.8+")
    print("• Scikit-learn (Machine Learning)")
    print("• NLTK (Natural Language Processing)")
    print("• TF-IDF Vectorization (Feature Extraction)")
    print("• Logistic Regression (Classification)")
    print("• Streamlit (Web Interface)")
    print("• Matplotlib/Seaborn (Visualization)")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("• src/preprocessing.py - Text preprocessing utilities")
    print("• src/train_model.py - Model training pipeline") 
    print("• src/evaluate_model.py - Model evaluation and metrics")
    print("• src/app.py - Streamlit web application")
    print("• models/ - Saved model files")
    print("• requirements.txt - Python dependencies")
    
    print("\n🚀 QUICK START:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run web app: streamlit run src/app.py")
    print("3. Or train model first: python src/train_model.py")
    print("4. Evaluate performance: python src/evaluate_model.py")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("• Model Accuracy: 85-90%")
    print("• Real-time emotion prediction")
    print("• Confidence scores for all emotions")
    print("• Interactive visualizations")
    print("• Comprehensive evaluation reports")
    
    print("\n🌟 APPLICATIONS:")
    print("• Social media sentiment monitoring")
    print("• Customer feedback analysis")
    print("• Chatbot emotion awareness")
    print("• Mental health tracking")
    print("• Content emotion analysis")
    
    print("\n" + "="*60)
    print("Ready to detect emotions! 🎭😊")
    print("="*60)

if __name__ == "__main__":
    # Demonstrate utility functions
    print_project_summary()
    
    # Create extended dataset
    print("\n📊 Creating extended sample dataset...")
    df = create_extended_sample_dataset()
    print(f"Dataset created with {len(df)} samples")
    
    # Analyze statistics
    stats = analyze_text_statistics(df)
    print(f"\n📈 Text Statistics:")
    print(f"• Average words per text: {stats['avg_words_per_text']:.1f}")
    print(f"• Average characters per text: {stats['avg_chars_per_text']:.1f}")
    print(f"• Text length range: {stats['min_words']}-{stats['max_words']} words")
    
    # Show emotion distribution
    print(f"\n🎭 Emotion Distribution:")
    emotion_dist = df['emotion'].value_counts()
    for emotion, count in emotion_dist.items():
        print(f"• {emotion.title()}: {count} samples")
    
    # Model information
    model_info = get_model_info()
    print(f"\n🤖 Model Information:")
    print(f"• Algorithm: {model_info['algorithm']}")
    print(f"• Feature Extraction: {model_info['feature_extraction']}")
    print(f"• Target Emotions: {', '.join(model_info['target_emotions'])}")
    print(f"• Expected Accuracy: {model_info['expected_accuracy']}")