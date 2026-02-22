import streamlit as st
from utils.session_state import init_session_state
from utils.api_client import analyze_emotion, check_backend_health
from utils.psychologists import get_matched_psychologists
st.set_page_config(
    page_title="Serenica",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

init_session_state()

#  Header
_, col_mid, _ = st.columns([1, 3, 1])
with col_mid:
    st.title("🌙 Serenica")
    st.caption("A safe space to reflect, understand, and heal.")

st.divider()

#  Tabs 
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Results", "💬 Chat"])



# TAB 1 — HOME / INPUT

with tab1:
    _, col, _ = st.columns([1, 3, 1])
    with col:
        # Backend health badge
        healthy = check_backend_health()
        if healthy:
            st.success("✅ AI models online and ready", icon=None)
        else:
            st.warning("⚠️ Backend offline — using fallback mode")

        st.markdown("## What's on your mind?")
        st.caption(
            "Share what you've been feeling lately. "
            "Serenica will listen, understand your emotions, "
            "and connect you with the right support."
        )

        # st.markdown("<br>", unsafe_allow_html=True)

        # Journal input
        user_text = st.text_area(
            "Your story",
            height=250,
            placeholder=(
                "Write freely — what's been weighing on your mind? "
                "There's no right or wrong way to share."
            ),
            label_visibility="collapsed",
            key="journal_text_area",
            value=st.session_state.user_text,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            reflect_clicked = st.button(
                "✨ Reflect",
                use_container_width=True,
                type="primary",
            )

        if reflect_clicked:
            if user_text.strip():
                st.session_state.user_text = user_text
                with st.spinner("Analyzing your reflection..."):
                    primary, confidence, probs, is_suicidal = analyze_emotion(user_text)

                st.session_state.analysis_done = True

                if is_suicidal:
                    st.error(
                        "🆘 We noticed signs of distress. Please reach out: "
                        "**Into The Light Indonesia** — 119 ext 8"
                    )

                st.success("✅ Analysis complete! Go to the **Results** tab to see your insights.")
                st.rerun()
            else:
                st.error("Please write something before reflecting.")

        st.markdown("<br><br>")
        st.caption("🔒 Your story is private. Serenica does not store personal data.")



# TAB 2 — RESULTS

with tab2:
    _, col, _ = st.columns([1, 4, 1])
    with col:
        if not st.session_state.analysis_result:
            st.info("📝 Write your story in the **Home** tab first, then come back here.")
            st.stop()

        result      = st.session_state.analysis_result
        primary     = result["primary_emotion"]
        confidence  = result["confidence"]
        probs       = st.session_state.emotion_probs
        suicidal    = st.session_state.get("suicidal_result", {})
        problem     = st.session_state.get("problem_result", {})

        #  Story preview 
        with st.expander("📖 Your story"):
            st.write(st.session_state.user_text)

        st.markdown("<br>", unsafe_allow_html=True)

        #  Metrics row 
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Primary Emotion", primary.title())
        with m2:
            st.metric("Confidence", f"{confidence*100:.0f}%")
        with m3:
            top_problem = max(problem, key=problem.get) if problem else "—"
            st.metric("Main Concern", top_problem.title())

        st.markdown("### 🔍 Analysis Summary")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            with st.container(border=True):
                st.markdown("**😶 Emotion Model**")
                if probs:
                    for label, score in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]:
                        st.write(f"{label.title()}: `{score*100:.1f}%`")

        with col_b:
            with st.container(border=True):
                st.markdown("**🧠 Problem Model**")
                if problem:
                    for label, score in sorted(problem.items(), key=lambda x: x[1], reverse=True)[:3]:
                        st.write(f"{label.title()}: `{score*100:.1f}%`")

        with col_c:
            with st.container(border=True):
                st.markdown("**⚠️ Risk Assessment**")
                if suicidal:
                    label = suicidal.get("label", "—").title()
                    conf  = suicidal.get("confidence", 0)
                    flag  = "🔴" if suicidal.get("is_suicidal") else "🟢"
                    st.write(f"{flag} **{label}**")
                    st.write(f"Confidence: `{conf*100:.1f}%`")
                else:
                    st.write("No data")

        # Crisis banner
        if suicidal.get("is_suicidal"):
            st.error(
                "🆘 **Crisis detected.** Please contact "
                "**Into The Light Indonesia** — 119 ext 8 "
                "or speak to a trusted person immediately."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        #  Emotion chart 
        if probs:
            import plotly.graph_objects as go

            st.subheader("📊 Emotion Distribution")
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels = [k.title() for k, _ in sorted_items]
            values = [v for _, v in sorted_items]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#667eea",
                text=[f"{v*100:.1f}%" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                height=max(250, len(labels) * 40),
                margin=dict(l=10, r=60, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, showticklabels=False,
                           range=[0, max(values) * 1.2]),
                yaxis=dict(showgrid=False, categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)

        #  Interpretation 
        interpretations = {
            "anxiety":      "You seem to be experiencing heightened worry. Acknowledging it is the first step.",
            "depression":   "Your reflection suggests low mood. These feelings are valid — support can help.",
            "stress":       "You appear to be under significant pressure. Self-reflection shows self-awareness.",
            "burnout":      "You may be experiencing prolonged fatigue. Rest and boundaries are essential.",
            "relationship": "Your story touches on interpersonal challenges. Seeking guidance is a sign of strength.",
            "grief":        "You seem to be processing loss. Grief takes many forms — all are valid.",
            "neutral":      "Your emotional state appears balanced. Continued self-reflection is valuable.",
        }
        interp = interpretations.get(primary, "Thank you for sharing your story.")
        st.info(f"💭 {interp}")

        # st.markdown("<br><br>", unsafe_allow_html=True)

        #  Psychologist matches 
        st.subheader("🧑‍⚕️ Recommended Specialists")
        st.caption("Matched based on your emotion profile, main concern, and risk assessment")

        from utils.psychologists import get_matched_psychologists

        top = get_matched_psychologists(
            emotion_probs=probs,
            problem_probs=problem,
            is_suicidal=suicidal.get("is_suicidal", False),
        )

        cols = st.columns(2)
        for i, psych in enumerate(top):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{psych['name']}**")
                    st.caption(psych["spec"])
                    st.caption(f"_{psych['bio']}_")
                    st.progress(psych["match"])
                    st.caption(f"Match score: **{psych['match']*100:.0f}%**")
                    if st.button("View Profile", key=f"psych_{i}"):
                        st.toast(f"Profile for {psych['name']} — coming soon!")


# TAB 3 — CHAT

with tab3:
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.subheader("💬 Chat with AI")

        # API key input
        api_key = st.text_input(
            "OpenAI or Anthropic API Key",
            type="password",
            placeholder="sk-...",
            value=st.session_state.api_key,
            key="chat_api_key_input",
        )
        if api_key:
            st.session_state.api_key = api_key

        if not st.session_state.api_key:
            st.warning("Enter your API key above to enable chat.")
            st.stop()

        st.caption("Continue exploring your thoughts with an AI companion.")

        # Chat history
        chat_box = st.container(height=400, border=True)
        with chat_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Input
        if prompt := st.chat_input("Type a message..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Context-aware system prompt if analysis exists
            context = ""
            if st.session_state.analysis_result:
                em = st.session_state.analysis_result["primary_emotion"]
                context = (
                    f"The user has been identified as experiencing '{em}'. "
                    "Respond with empathy and psychological support."
                )

            # Try calling AI API
            try:
                from utils.chat_client import get_ai_response
                ai_reply = get_ai_response(
                    api_key=st.session_state.api_key,
                    messages=st.session_state.chat_messages,
                    system_context=context,
                )
            except Exception as e:
                ai_reply = f"(Chat unavailable: {e})"

            st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
            st.rerun()

        if st.button("🗑️ Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()