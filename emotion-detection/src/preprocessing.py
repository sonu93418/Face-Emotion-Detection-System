"""
Text Preprocessing Module for Emotion Detection
Contains functions for cleaning, tokenizing, and preprocessing text data
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with required NLTK downloads"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = None
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text by removing URLs, mentions, hashtags, and special characters
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into individual words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their root form
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column, target_column=None):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            target_column (str): Name of target column (optional)
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Starting text preprocessing...")
        
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Preprocess text column
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.preprocess_text)
        
        # Remove empty texts
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
        
        print(f"Preprocessing completed. {len(processed_df)} samples remaining.")
        
        return processed_df

def create_sample_dataset():
    """
    Create a sample emotion dataset for demonstration
    This simulates the Twitter Emotion Dataset from Kaggle
    """
    sample_data = {
        'text': [
            "I'm so happy today! The weather is beautiful and I feel amazing!",
            "This is really frustrating. Nothing is going right today.",
            "I'm scared about the upcoming exam. What if I fail?",
            "Feeling sad about the news. My heart goes out to everyone affected.",
            "Just had an amazing dinner with friends. Life is good!",
            "I can't believe this happened to me. I'm so angry right now!",
            "The movie was okay. Nothing special but not bad either.",
            "I love spending time with my family during holidays.",
            "This traffic is making me so mad. I'm going to be late!",
            "Worried about my job interview tomorrow. Hope it goes well.",
            "Had the best day ever at the amusement park!",
            "Feeling lonely today. Wish I had someone to talk to.",
            "The concert was incredible! The band was amazing!",
            "I hate when people don't listen to what I'm saying.",
            "Nervous about my presentation but I think I'm prepared.",
            "Absolutely thrilled about my promotion at work!",
            "This rainy weather is making me feel down.",
            "Terrified of flying but I have to take this trip.",
            "Just a normal day at the office. Nothing exciting.",
            "Overjoyed to hear that my friend is getting married!"
        ],
        'emotion': [
            'joy', 'anger', 'fear', 'sadness', 'joy', 'anger', 'neutral', 'joy',
            'anger', 'fear', 'joy', 'sadness', 'joy', 'anger', 'fear', 'joy',
            'sadness', 'fear', 'neutral', 'joy'
        ]
    }
    
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    # Test the preprocessing module
    preprocessor = TextPreprocessor()
    
    # Create sample dataset
    df = create_sample_dataset()
    print("Sample dataset created with", len(df), "samples")
    
    # Test preprocessing
    sample_text = "I'm so excited about this new project! Can't wait to get started! 😊"
    cleaned = preprocessor.preprocess_text(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Preprocessed: {cleaned}")
    
    # Preprocess entire dataframe
    processed_df = preprocessor.preprocess_dataframe(df, 'text', 'emotion')
    print(f"\nProcessed dataframe shape: {processed_df.shape}")
    print("\nFirst few preprocessed texts:")
    for i in range(3):
        print(f"Original: {processed_df.iloc[i]['text']}")
        print(f"Cleaned: {processed_df.iloc[i]['cleaned_text']}")
        print(f"Emotion: {processed_df.iloc[i]['emotion']}\n")