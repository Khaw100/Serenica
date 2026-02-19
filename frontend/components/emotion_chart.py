import streamlit as st
import plotly.graph_objects as go

def render_emotion_chart():
    """Display emotion distribution bar chart"""
    
    if not st.session_state.emotion_probs:
        st.warning("No emotion data available")
        return
    
    st.subheader("📊 Emotion Distribution")
    
    # Prepare data
    emotions = list(st.session_state.emotion_probs.keys())
    probabilities = list(st.session_state.emotion_probs.values())
    
    # Sort by probability (highest first)
    sorted_data = sorted(
        zip(emotions, probabilities), 
        key=lambda x: x[1], 
        reverse=True
    )
    emotions_sorted = [x[0].title() for x in sorted_data]
    probs_sorted = [x[1] for x in sorted_data]
    
    # Soft color palette
    colors = [
        '#667eea',  # Primary purple
        '#764ba2',  # Deep purple
        '#8b5cf6',  # Violet
        '#a78bfa',  # Light violet
        '#c4b5fd',  # Lighter violet
        '#ddd6fe',  # Very light violet
        '#ede9fe'   # Palest violet
    ]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=probs_sorted,
        y=emotions_sorted,
        orientation='h',
        marker=dict(
            color=colors[:len(emotions_sorted)],
            line=dict(color='rgba(255,255,255,0.8)', width=2)
        ),
        text=[f"{p*100:.1f}%" for p in probs_sorted],
        textposition='outside',
    ))
    
    # Layout styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="sans-serif",
            size=12,
            color="#374151"
        ),
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, max(probs_sorted) * 1.15]
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder='total ascending'
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)