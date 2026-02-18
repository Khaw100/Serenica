import numpy as np
from datetime import datetime
import streamlit as st

def analyze_emotion(text):
    """
    Mock emotion analysis - TODO: Replace with BiLSTM model
    """
    emotions = ["anxiety", "depression", "stress", "burnout", 
                "relationship", "grief", "neutral"]
    
    # Generate mock probabilities
    np.random.seed(len(text))
    probs = np.random.dirichlet(np.ones(7) * 2)
    
    emotion_probs = dict(zip(emotions, probs))
    primary_emotion = max(emotion_probs, key=emotion_probs.get)
    confidence = emotion_probs[primary_emotion]
    
    # Store in session state
    st.session_state.emotion_probs = emotion_probs
    st.session_state.analysis_result = {
        'primary_emotion': primary_emotion,
        'confidence': confidence,
        'timestamp': datetime.now()
    }
    
    # Add to history
    st.session_state.history.append({
        'date': datetime.now(),
        'emotion': primary_emotion,
        'confidence': confidence
    })
    
    return primary_emotion, confidence, emotion_probs