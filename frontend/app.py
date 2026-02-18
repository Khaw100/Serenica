import streamlit as st
from styles.custom_css import load_css
from utils.session_state import init_session_state, go_to_results
from utils.ai_model import analyze_emotion
from components.header import render_header
from components.api_key import render_api_key_input
from components.journal_input import render_journal_input
from components.chat import render_chat_interface

st.set_page_config(
    page_title="Serenica",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# load_css()
init_session_state()

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    render_header()
    render_api_key_input()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    def on_journal_submit(text):
        analyze_emotion(text)
        go_to_results()
        st.rerun()
    
    render_journal_input(on_submit_callback=on_journal_submit)
    
    # Chat interface (hanya muncul kalau ada API key)
    render_chat_interface()
    
    # Privacy footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("🔒 Your story is private. Serenica does not store personal data.")