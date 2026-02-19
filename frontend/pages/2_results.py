import streamlit as st
from styles.custom_css import load_css
from utils.session_state import init_session_state
from components.header import render_header
from components.emotion_metrics import render_emotion_metrics
from components.emotion_chart import render_emotion_chart
from components.interpretation import render_interpretation
from components.psychologist_card import render_psychologist_cards

# Page config
st.set_page_config(
    page_title="Serenica - Results",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# load_css()
init_session_state()

# Check if analysis exists
if not st.session_state.analysis_result:
    st.warning("No analysis available. Please write your story first.")
    if st.button("← Back to Input"):
        st.switch_page("pages/1_input.py")
    st.stop()

# Center layout
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    render_header(
        title="Your Emotional Insight",
        subtitle="This reflection is based on your story."
    )
    
    # Story preview (collapsible)
    with st.expander("📖 See your story"):
        st.write(st.session_state.user_text)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Emotion analysis
    render_emotion_metrics()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_emotion_chart()
    
    st.markdown("<br>", unsafe_allow_html=True)
    render_interpretation()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_psychologist_cards()
    
    # Navigation button
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3 = st.columns([2, 1, 2])
    with col_nav2:
        if st.button("✍️ Write Another Story", use_container_width=True):
            st.session_state.user_text = ''
            st.session_state.analysis_result = None
            st.switch_page("pages/1_input.py")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🔒 Your story is private. Serenica does not store personal data.")