"""
Streamlit Web Application for Real-time Face Emotion Detection
Uses webcam to detect facial emotions in real-time through the browser
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import queue
import time

# Import our face emotion detector
from face_emotion_detector import FaceEmotionDetector

# Page configuration
st.set_page_config(
    page_title="🎭 Face Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .instruction-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = FaceEmotionDetector()
        self.emotion_history = []
        self.frame_count = 0
        self.lock = threading.Lock()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame for emotion detection
        processed_img, emotions = self.detector.process_frame(img)
        
        # Store emotion data
        with self.lock:
            self.frame_count += 1
            if emotions:
                for emotion_data in emotions:
                    self.emotion_history.append({
                        'timestamp': datetime.now(),
                        'frame': self.frame_count,
                        'emotion': emotion_data['emotion'],
                        'confidence': emotion_data['confidence'],
                        'all_confidences': emotion_data['all_confidences']
                    })
        
        return processed_img
    
    def get_emotion_history(self):
        with self.lock:
            return self.emotion_history.copy()

# Initialize session state
if 'emotion_stats' not in st.session_state:
    st.session_state.emotion_stats = {}
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

def create_emotion_chart(emotion_data):
    """Create emotion distribution chart"""
    if not emotion_data:
        return None
    
    df = pd.DataFrame(emotion_data)
    emotion_counts = df['emotion'].value_counts()
    
    colors = {
        'Happy': '#2ECC71',
        'Sad': '#3498DB', 
        'Angry': '#E74C3C',
        'Fear': '#F39C12',
        'Surprise': '#9B59B6',
        'Disgust': '#8E44AD',
        'Neutral': '#95A5A6'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            marker_color=[colors.get(emotion, '#BDC3C7') for emotion in emotion_counts.index],
            text=emotion_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Detection Statistics",
        xaxis_title="Emotions",
        yaxis_title="Number of Detections",
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_timeline(emotion_data):
    """Create timeline of emotion confidence"""
    if not emotion_data:
        return None
    
    df = pd.DataFrame(emotion_data)
    
    fig = go.Figure()
    
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        fig.add_trace(go.Scatter(
            x=emotion_df['timestamp'],
            y=emotion_df['confidence'],
            mode='lines+markers',
            name=emotion,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Emotion Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence Score",
        height=400,
        showlegend=True
    )
    
    return fig

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

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Real-time Face Emotion Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎥 **Live Webcam Emotion Recognition**
    Experience the power of **Computer Vision** and **Deep Learning** to detect emotions from your facial expressions in real-time!
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 **Detection Settings**")
        
        # WebRTC settings
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
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
        1. **Face Detection**: OpenCV detects faces in real-time
        2. **Feature Extraction**: CNN processes facial features
        3. **Emotion Classification**: AI predicts emotions
        4. **Real-time Display**: Results shown with confidence scores
        """)
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.emotion_stats = {}
            st.session_state.total_detections = 0
            st.success("Statistics reset!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 **Live Camera Feed**")
        
        # Instructions
        st.markdown("""
        <div class="instruction-box">
            <h4>📋 Instructions:</h4>
            <ul>
                <li>🎥 Allow camera access when prompted</li>
                <li>👤 Position your face clearly in the camera</li>
                <li>😊 Try different facial expressions</li>
                <li>📊 Watch real-time emotion detection</li>
                <li>📈 View statistics in the right panel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # WebRTC video streamer
        ctx = webrtc_streamer(
            key="face-emotion-detection",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
        )
        
        # Display current detection status
        if ctx.video_transformer:
            st.markdown("### 🔍 **Detection Status**")
            
            # Get recent emotion data
            emotion_history = ctx.video_transformer.get_emotion_history()
            
            if emotion_history:
                # Show latest detection
                latest_detection = emotion_history[-1]
                emotion = latest_detection['emotion']
                confidence = latest_detection['confidence']
                emoji = get_emotion_emoji(emotion)
                color = get_emotion_color(emotion)
                
                st.markdown(f"""
                <div class="emotion-card">
                    <h2 style="margin: 0; text-align: center;">
                        {emoji} {emotion}
                    </h2>
                    <h3 style="margin: 0.5rem 0; text-align: center;">
                        Confidence: {confidence:.1%}
                    </h3>
                    <p style="margin: 0; text-align: center; opacity: 0.8;">
                        Last detected: {latest_detection['timestamp'].strftime('%H:%M:%S')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence breakdown for latest detection
                st.markdown("#### 📊 **Confidence Breakdown**")
                confidences = latest_detection['all_confidences']
                
                for emo, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
                    progress_color = get_emotion_color(emo)
                    emoji = get_emotion_emoji(emo)
                    st.write(f"{emoji} **{emo}**: {conf:.1%}")
                    st.progress(conf)
            else:
                st.info("👤 No faces detected yet. Make sure your face is visible in the camera!")
    
    with col2:
        st.markdown("### 📈 **Real-time Statistics**")
        
        if ctx.video_transformer:
            emotion_history = ctx.video_transformer.get_emotion_history()
            
            if emotion_history:
                # Basic statistics
                total_detections = len(emotion_history)
                unique_emotions = len(set([d['emotion'] for d in emotion_history]))
                avg_confidence = np.mean([d['confidence'] for d in emotion_history])
                
                # Display stats cards
                st.markdown(f"""
                <div class="stats-card">
                    <h3>🎯 Total Detections</h3>
                    <h2>{total_detections}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stats-card">
                    <h3>🎭 Emotions Detected</h3>
                    <h2>{unique_emotions}/7</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stats-card">
                    <h3>📊 Avg Confidence</h3>
                    <h2>{avg_confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Emotion distribution chart
                if len(emotion_history) > 5:
                    fig = create_emotion_chart(emotion_history)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recent detections
                st.markdown("#### 🕒 **Recent Detections**")
                recent_data = emotion_history[-10:]  # Last 10 detections
                
                for detection in reversed(recent_data):
                    emoji = get_emotion_emoji(detection['emotion'])
                    time_str = detection['timestamp'].strftime('%H:%M:%S')
                    
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; border-left: 3px solid {get_emotion_color(detection['emotion'])};">
                        {emoji} <strong>{detection['emotion']}</strong> ({detection['confidence']:.1%}) - {time_str}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export data option
                if st.button("📥 Export Detection Data"):
                    df = pd.DataFrame(emotion_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"emotion_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("📊 Statistics will appear once faces are detected!")
        else:
            st.info("🎥 Start the camera to begin emotion detection!")
    
    # Additional features
    st.markdown("---")
    st.markdown("### 🚀 **Features & Capabilities**")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        #### 🎯 **Detection Features**
        - Real-time face detection
        - 7 emotion categories
        - Confidence scoring
        - Multiple face support
        """)
    
    with features_col2:
        st.markdown("""
        #### 📊 **Analytics**
        - Live statistics
        - Emotion distribution
        - Confidence tracking
        - Data export
        """)
    
    with features_col3:
        st.markdown("""
        #### 🔧 **Technology**
        - OpenCV face detection
        - TensorFlow CNN model
        - Real-time processing
        - Browser-based webcam
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px;">
        <h3>🎭 Real-time Face Emotion Detection System</h3>
        <p><strong>Powered by Computer Vision & Deep Learning</strong></p>
        <p>🎥 OpenCV • 🧠 TensorFlow • 🌐 Streamlit WebRTC • 📊 Real-time Analytics</p>
        <p><em>Experience the future of emotion recognition technology</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()