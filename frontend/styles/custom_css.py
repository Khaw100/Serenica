import streamlit as st

def load_css():
    st.markdown("""
    <style>
        /* Background color */
        [data-testid="stAppViewContainer"] {
            background-color: #f8f9fa;
        }

        html, body, [class*="css"]  {
            color: black !important;
        }
                
        /* Hide menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Button styling */
        .stButton > button {
            border-radius: 24px;
            padding: 12px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 500;
            border: none;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)