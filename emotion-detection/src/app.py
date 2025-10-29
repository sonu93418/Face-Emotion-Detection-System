"""
Streamlit Web Application for Emotion Detection
Interactive interface for real-time emotion prediction from text input
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_model import EmotionClassifier
    from preprocessing import TextPreprocessor
except ImportError:
    st.error("Required modules not found. Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        padding: 0.25rem;
        margin: 0.5rem 0;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Load the trained emotion detection model
    """
    try:
        classifier = EmotionClassifier()
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        classifier.load_model(model_path)
        return classifier
    except FileNotFoundError:
        # If no pre-trained model exists, train a new one
        st.warning("No pre-trained model found. Training a new model...")
        classifier = EmotionClassifier()
        
        # Load and train on sample data
        df = classifier.load_data(use_sample=True)
        processed_df = classifier.preprocess_data(df)
        X, y = classifier.prepare_features(processed_df)
        
        # Simple train without test split for demo
        classifier.train_model(X, y)
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        classifier.save_model(model_path)
        
        st.success("Model trained and saved successfully!")
        return classifier

def create_emotion_chart(confidence_scores):
    """
    Create a bar chart for emotion confidence scores
    """
    emotions = list(confidence_scores.keys())
    scores = [confidence_scores[emotion] * 100 for emotion in emotions]
    
    # Create color map for emotions
    color_map = {
        'joy': '#28a745',
        'anger': '#dc3545', 
        'sadness': '#6c757d',
        'fear': '#fd7e14',
        'neutral': '#17a2b8'
    }
    
    colors = [color_map.get(emotion.lower(), '#007bff') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=scores,
            marker_color=colors,
            text=[f'{score:.1f}%' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis_title="Emotions",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def get_emotion_emoji(emotion):
    """
    Get emoji representation for emotions
    """
    emoji_map = {
        'joy': '😊',
        'anger': '😠',
        'sadness': '😢',
        'fear': '😨',
        'neutral': '😐'
    }
    return emoji_map.get(emotion.lower(), '🤔')

def get_emotion_description(emotion):
    """
    Get description for emotions
    """
    descriptions = {
        'joy': "Positive emotion characterized by happiness, contentment, and satisfaction.",
        'anger': "Negative emotion involving feelings of hostility, resentment, and frustration.",
        'sadness': "Negative emotion characterized by feelings of disadvantage, loss, and helplessness.",
        'fear': "Emotion induced by perceived danger or threat, causing anxiety and worry.",
        'neutral': "Balanced emotional state without strong positive or negative feelings."
    }
    return descriptions.get(emotion.lower(), "Unknown emotion detected.")

def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<h1 class="main-header">🎭 Emotion Detection from Text</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses **Machine Learning** to detect emotions in text using **Natural Language Processing**.
    Simply enter any text or tweet below, and the system will predict the emotional sentiment.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About the Model")
        st.info("""
        **Technologies Used:**
        - Python & Scikit-learn
        - TF-IDF Vectorization
        - Logistic Regression
        - NLTK for text preprocessing
        - Streamlit for web interface
        
        **Emotion Categories:**
        - 😊 Joy
        - 😠 Anger  
        - 😢 Sadness
        - 😨 Fear
        - 😐 Neutral
        """)
        
        st.header("🎯 Model Performance")
        st.success("Target Accuracy: 85-90%")
        
        if st.button("📈 View Model Statistics"):
            st.balloons()
    
    # Load model
    with st.spinner("Loading emotion detection model..."):
        model = load_model()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Text Input")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ("Type your own text", "Select sample text"),
            horizontal=True
        )
        
        if input_method == "Type your own text":
            user_text = st.text_area(
                "Enter text to analyze:",
                placeholder="Type or paste your text here... (e.g., 'I'm so excited about this new opportunity!')",
                height=120,
                key="user_input"
            )
        else:
            # Sample texts for testing
            sample_texts = [
                "I'm absolutely thrilled about this amazing opportunity!",
                "This situation is making me extremely frustrated and angry!",
                "I'm really worried and scared about what might happen next.",
                "Feeling quite sad and down today, nothing seems to go right.",
                "It's just another ordinary day at work, nothing special happening.",
                "I can't believe how wonderful this surprise party was!",
                "This unfair treatment is making me furious!",
                "The thought of that upcoming surgery terrifies me.",
                "I'm heartbroken about the loss of my dear friend.",
                "Weather is okay today, not too hot, not too cold."
            ]
            
            selected_text = st.selectbox(
                "Choose a sample text:",
                options=sample_texts,
                key="sample_select"
            )
            user_text = selected_text
        
        # Predict button
        if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing emotion..."):
                    try:
                        # Get prediction
                        predicted_emotion, confidence_scores = model.predict_emotion(user_text)
                        
                        # Store results in session state
                        st.session_state['last_prediction'] = {
                            'text': user_text,
                            'emotion': predicted_emotion,
                            'confidence': confidence_scores,
                            'timestamp': datetime.now()
                        }
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.stop()
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("📊 Results")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            
            # Display main result
            emoji = get_emotion_emoji(pred['emotion'])
            st.markdown(f"""
            <div class="emotion-card">
                <h2 style="color: #1f77b4; margin: 0;">
                    {emoji} {pred['emotion'].title()}
                </h2>
                <p style="margin: 0.5rem 0;">
                    Confidence: <strong>{pred['confidence'][pred['emotion']]:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Description
            description = get_emotion_description(pred['emotion'])
            st.info(description)
            
            # Confidence chart
            st.plotly_chart(
                create_emotion_chart(pred['confidence']), 
                use_container_width=True
            )
            
            # Detailed scores
            st.subheader("📋 Detailed Scores")
            for emotion, score in sorted(pred['confidence'].items(), 
                                       key=lambda x: x[1], reverse=True):
                emoji = get_emotion_emoji(emotion)
                st.write(f"{emoji} **{emotion.title()}**: {score:.1%}")
        
        else:
            st.info("👆 Enter text above and click 'Analyze Emotion' to see results!")
    
    # History section
    st.header("📝 Analysis History")
    
    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        
        history_df = pd.DataFrame([{
            'Timestamp': pred['timestamp'].strftime('%H:%M:%S'),
            'Text': pred['text'][:50] + '...' if len(pred['text']) > 50 else pred['text'],
            'Predicted Emotion': pred['emotion'].title(),
            'Confidence': f"{pred['confidence'][pred['emotion']]:.1%}"
        }])
        
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No analysis history yet. Start by analyzing some text!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <h4>🔬 Machine Learning Emotion Detection System</h4>
        <p>Built with Python • Scikit-learn • NLTK • Streamlit</p>
        <p><em>Accurately detecting human emotions from text using AI</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()