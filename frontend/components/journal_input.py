import streamlit as st

def render_journal_input(on_submit_callback=None):
    with st.container():
        st.subheader("What's on your mind?")

        user_input = st.text_area(
            "Your story",
            height = 250,
            placeholder="What's been on your mind lately? There's no right",
            label_visibility="collapsed",
            key="journal_text_area"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            if st.button("✨ Reflect", use_container_width=True):
                if user_input.strip():
                    # Save to session state
                    st.session_state.user_text = user_input
                    
                    # Call callback if provided
                    if on_submit_callback:
                        on_submit_callback(user_input)
                    
                    return user_input
                else:
                    st.error("Please share something before reflecting")
                    return None
        
        return None