import streamlit as st


def init_session_state():
    defaults = {
        "user_text": "",
        "api_key": "",
        "analysis_result": None,
        "emotion_probs": {},
        "suicidal_result": {},
        "problem_result": {},
        "chat_messages": [],
        "history": [],
        "analysis_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val