import streamlit as st

def render_emotion_metrics():
    if not st.session_state.analysis_result:
        st.warning("No analysis available")
        return
    
    primary = st.session_state.analysis_result['primary_emotion']
    confidence = st.session_state.analysis_result['confidence']
    
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.metric(
            label="Primary Emotion",
            value=primary.title(),
            help="The dominant emotion detected in your reflection"
        )
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.0f}%",
            help="How certain the AI is about this classification"
        )
    
    with col_metric2:
        st.write("")
        st.write("")
        st.progress(confidence)
        st.caption(f"Analysis confidence: {confidence*100:.1f}%")