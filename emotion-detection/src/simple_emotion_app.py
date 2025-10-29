"""
Simple Streamlit Web Application for Face Emotion Detection
Alternative approach using file upload and image processing
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Import our face emotion detector
from face_emotion_detector import FaceEmotionDetector

# Page configuration
st.set_page_config(
    page_title="🎭 Face Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stats-card {
        background: white;
        border: 2px solid #e1e8ed;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def get_emotion_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😠',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢',
        'Neutral': '😐'
    }
    return emoji_map.get(emotion, '🤔')

def get_emotion_color(emotion):
    """Get color for emotion"""
    color_map = {
        'Happy': '#2ECC71',
        'Sad': '#3498DB',
        'Angry': '#E74C3C',
        'Fear': '#F39C12',
        'Surprise': '#9B59B6',
        'Disgust': '#8E44AD',
        'Neutral': '#95A5A6'
    }
    return color_map.get(emotion, '#BDC3C7')

@st.cache_resource
def load_detector():
    """Load face emotion detector"""
    return FaceEmotionDetector()

def process_image(image, detector):
    """Process uploaded image for emotion detection"""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process frame for emotion detection
    processed_img, emotions = detector.process_frame(img_bgr)
    
    # Convert result back to RGB for display
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    return processed_img_rgb, emotions

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Face Emotion Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 📸 **Image-based Face Emotion Recognition**
    Upload an image or take a photo to detect emotions from facial expressions using **Computer Vision** and **Deep Learning**!
    """)
    
    # Load detector
    detector = load_detector()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎭 **Supported Emotions**")
        emotions_info = [
            ("😊 Happy", "#2ECC71"),
            ("😢 Sad", "#3498DB"),
            ("😠 Angry", "#E74C3C"),
            ("😨 Fear", "#F39C12"),
            ("😲 Surprise", "#9B59B6"),
            ("🤢 Disgust", "#8E44AD"),
            ("😐 Neutral", "#95A5A6")
        ]
        
        for emoji_emotion, color in emotions_info:
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0; border-left: 3px solid {color};">
                <strong>{emoji_emotion}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ℹ️ **How it Works**")
        st.info("""
        1. **Upload Image**: Choose an image with faces
        2. **Face Detection**: OpenCV detects faces automatically
        3. **Feature Extraction**: CNN processes facial features
        4. **Emotion Classification**: AI predicts emotions with confidence scores
        5. **Visual Results**: See annotated image with emotion labels
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["📸 Upload Image", "📷 Camera Capture"])
    
    with tab1:
        st.markdown("### 📸 **Upload an Image**")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing faces to detect emotions"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🖼️ **Original Image**")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("#### 🎭 **Emotion Detection Results**")
                
                # Process image
                with st.spinner("🔍 Processing image for emotion detection..."):
                    processed_img, emotions = process_image(image, detector)
                
                # Display processed image
                st.image(processed_img, caption="Processed Image with Emotion Detection", use_column_width=True)
            
            # Display results
            if emotions:
                st.markdown("### 📊 **Detection Results**")
                
                for i, emotion_data in enumerate(emotions):
                    emotion = emotion_data['emotion']
                    confidence = emotion_data['confidence']
                    emoji = get_emotion_emoji(emotion)
                    color = get_emotion_color(emotion)
                    
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h3 style="margin: 0;">
                            Face {i+1}: {emoji} {emotion} ({confidence:.1%} confidence)
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence breakdown
                    st.markdown(f"**Confidence Breakdown for Face {i+1}:**")
                    confidences = emotion_data['all_confidences']
                    
                    for emo, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
                        emoji = get_emotion_emoji(emo)
                        st.write(f"{emoji} **{emo}**: {conf:.1%}")
                        st.progress(conf)
                    
                    st.markdown("---")
                
                # Summary statistics
                st.markdown("### 📈 **Summary Statistics**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h3>👥 Faces Detected</h3>
                        <h2>{len(emotions)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_confidence = np.mean([e['confidence'] for e in emotions])
                    st.markdown(f"""
                    <div class="stats-card">
                        <h3>📊 Avg Confidence</h3>
                        <h2>{avg_confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    dominant_emotion = max(emotions, key=lambda x: x['confidence'])['emotion']
                    emoji = get_emotion_emoji(dominant_emotion)
                    st.markdown(f"""
                    <div class="stats-card">
                        <h3>🎯 Dominant Emotion</h3>
                        <h2>{emoji} {dominant_emotion}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.warning("😕 No faces detected in the image. Please try uploading an image with clear, visible faces.")
    
    with tab2:
        st.markdown("### 📷 **Camera Capture**")
        
        st.info("""
        📱 **For camera access:**
        - Click "Take a picture" below
        - Allow camera permissions when prompted
        - Capture your photo for emotion detection
        """)
        
        # Camera input
        camera_image = st.camera_input("Take a picture for emotion detection")
        
        if camera_image is not None:
            # Process camera image
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 **Captured Image**")
                st.image(image, caption="Camera Capture", use_column_width=True)
            
            with col2:
                st.markdown("#### 🎭 **Emotion Detection Results**")
                
                # Process image
                with st.spinner("🔍 Analyzing your expression..."):
                    processed_img, emotions = process_image(image, detector)
                
                # Display processed image
                st.image(processed_img, caption="Emotion Detection Results", use_column_width=True)
            
            # Display results (same as upload tab)
            if emotions:
                st.markdown("### 📊 **Your Emotion Analysis**")
                
                for i, emotion_data in enumerate(emotions):
                    emotion = emotion_data['emotion']
                    confidence = emotion_data['confidence']
                    emoji = get_emotion_emoji(emotion)
                    
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h3 style="margin: 0; text-align: center;">
                            {emoji} You look {emotion.lower()}! ({confidence:.1%} confidence)
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence breakdown
                    st.markdown("**How confident are we about each emotion?**")
                    confidences = emotion_data['all_confidences']
                    
                    for emo, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
                        emoji = get_emotion_emoji(emo)
                        st.write(f"{emoji} **{emo}**: {conf:.1%}")
                        st.progress(conf)
            else:
                st.warning("😕 No faces detected in your photo. Please take another picture with your face clearly visible.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px;">
        <h3>🎭 Face Emotion Detection System</h3>
        <p><strong>Powered by Computer Vision & Deep Learning</strong></p>
        <p>🎥 OpenCV • 🧠 TensorFlow • 📊 Advanced Analytics</p>
        <p><em>Experience the power of AI emotion recognition</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()