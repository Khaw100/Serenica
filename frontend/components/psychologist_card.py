import streamlit as st

def get_mock_psychologists(emotion):
    """
    Get mock psychologist data
    TODO: Replace with real database/API
    """
    psychologists = [
        {
            'name': 'Dr. Sarah Chen',
            'specialization': 'Anxiety & Stress Management',
            'match': 0.94,
            'image': 'https://via.placeholder.com/150/667eea/ffffff?text=SC'
        },
        {
            'name': 'Dr. Michael Rodriguez',
            'specialization': 'Cognitive Behavioral Therapy',
            'match': 0.89,
            'image': 'https://via.placeholder.com/150/764ba2/ffffff?text=MR'
        },
        {
            'name': 'Dr. Emily Johnson',
            'specialization': 'Depression & Mood Disorders',
            'match': 0.85,
            'image': 'https://via.placeholder.com/150/8b5cf6/ffffff?text=EJ'
        },
        {
            'name': 'Dr. David Kim',
            'specialization': 'Relationship Counseling',
            'match': 0.78,
            'image': 'https://via.placeholder.com/150/a78bfa/ffffff?text=DK'
        }
    ]
    
    return sorted(psychologists, key=lambda x: x['match'], reverse=True)[:4]

def render_psychologist_cards():
    """Display psychologist matching cards in 2-column grid"""
    
    if not st.session_state.analysis_result:
        return
    
    primary = st.session_state.analysis_result['primary_emotion']
    
    st.subheader("🧑‍⚕️ Your Psychologist Matches")
    st.caption("Based on your emotional profile, here are specialists who can help")
    
    psychologists = get_mock_psychologists(primary)
    
    # Display in 2-column grid
    for i in range(0, len(psychologists), 2):
        col1, col2 = st.columns(2)
        
        # First card in row
        with col1:
            if i < len(psychologists):
                psych = psychologists[i]
                with st.container():
                    st.image(psych['image'], width=120)
                    st.markdown(f"**{psych['name']}**")
                    st.caption(psych['specialization'])
                    st.progress(psych['match'])
                    st.caption(f"Match: {psych['match']*100:.0f}%")
                    
                    if st.button("View Profile", key=f"btn_{i}"):
                        st.info(f"Profile view for {psych['name']} - Coming soon!")
        
        # Second card in row
        with col2:
            if i + 1 < len(psychologists):
                psych = psychologists[i + 1]
                with st.container():
                    st.image(psych['image'], width=120)
                    st.markdown(f"**{psych['name']}**")
                    st.caption(psych['specialization'])
                    st.progress(psych['match'])
                    st.caption(f"Match: {psych['match']*100:.0f}%")
                    
                    if st.button("View Profile", key=f"btn_{i+1}"):
                        st.info(f"Profile view for {psych['name']} - Coming soon!")