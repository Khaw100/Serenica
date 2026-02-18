import streamlit as st

def render_header(title="Serenica", subtitle="A safe space to be heard."):
    """Render page header"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center layout dengan columns
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.title(f"🌙 {title}")
        st.caption(subtitle)
        st.markdown("<br>", unsafe_allow_html=True)