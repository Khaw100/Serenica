import streamlit as st

def render_api_key_input():
    """Render API key input section"""
    
    with st.container():
        st.subheader("🔑 Connect your AI")
        
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your OpenAI or Anthropic API key",
            value=st.session_state.api_key,
            label_visibility="collapsed",
            key="api_key_input"
        )
        
        # Update session state
        if api_key:
            st.session_state.api_key = api_key
            st.success("✓ AI connected successfully")
        else:
            st.warning("⚠️ Chat features disabled without API key")
        
        return api_key