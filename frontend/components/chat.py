import streamlit as st

def render_chat_interface():
    """Render AI chat interface (only if API key exists)"""

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💬 Chat with AI")

    if not st.session_state.api_key:
        return st.warning("API KEY REQUIRED")

    st.caption("Have a conversation if you'd like to explore your thoughts further")
    chat_box = st.container(height=400, border=True)

    with chat_box:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if prompt := st.chat_input("Type a messasge..."):
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })

        ai_response = "I hear you. This is a mock response. (Integrate your AI API here)"

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ai_response
        })

        st.rerun()
