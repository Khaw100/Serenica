import streamlit as st

def get_interpretation_text(emotion, confidence):
    """Get supportive interpretation based on emotion"""
    
    interpretations = {
        'anxiety': "You seem to be experiencing heightened worry and mental restlessness. This is a natural response to stress, and acknowledging it is an important first step.",
        'depression': "Your reflection suggests feelings of low mood and reduced energy. Remember that these feelings are valid, and professional support can make a meaningful difference.",
        'stress': "You appear to be under significant pressure. Taking time to reflect like this shows self-awareness and care for your wellbeing.",
        'burnout': "You may be experiencing prolonged mental and physical fatigue. Rest and boundary-setting are essential for recovery.",
        'relationship': "Your story indicates interpersonal challenges. Relationship dynamics can deeply affect our emotional state, and it's okay to seek guidance.",
        'grief': "You seem to be processing loss or significant change. Grief is a personal journey, and there's no 'right' way to feel.",
        'neutral': "Your emotional state appears balanced. Maintaining this equilibrium through continued self-reflection is valuable."
    }
    
    return interpretations.get(emotion, "Thank you for sharing your story.")

def render_interpretation():
    """Display emotional state interpretation"""
    
    if not st.session_state.analysis_result:
        return
    
    primary = st.session_state.analysis_result['primary_emotion']
    confidence = st.session_state.analysis_result['confidence']
    
    interpretation = get_interpretation_text(primary, confidence)
    
    st.info(f"💭 {interpretation}")