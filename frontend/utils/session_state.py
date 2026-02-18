import streamlit as st

def init_session_state():
    """Initialize all session state variables"""
    
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
    
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ''
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    if 'emotion_probs' not in st.session_state:
        st.session_state.emotion_probs = {}
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'history' not in st.session_state:
        st.session_state.history = []

# Navigation functions
def go_to_results():
    st.session_state.page = 'results'

def go_to_input():
    st.session_state.page = 'input'
    st.session_state.user_text = ''
    st.session_state.analysis_result = None