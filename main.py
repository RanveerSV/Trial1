import streamlit as st
import pandas as pd
import os
import random
from helper import find_most_likely, parse_chat

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="Likely To... Analyzer", page_icon="🎉", layout="wide")

# Injecting Custom CSS for a "Fun" vibe
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Card-like containers for results */
    .result-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 30px;
        border: none;
        height: 3em;
        background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        color: #4a4a4a !important;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 10px 20px rgba(0,0,0,0.2);
    }

    /* Titles and text */
    h1, h2, h3 {
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar for Upload (Keeps main screen clean)
with st.sidebar:
    st.title("Settings ⚙️")
    uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])
    st.divider()
    st.info("💡 **Pro-tip:** Use the 'Surprise Me' button if you can't think of a prompt!")

# 3. Main Hero Section
st.title("🏆 Who is Most Likely To...?")
st.write("Upload your chat and let the AI settle the debate once and for all!")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_filename = "temp_chat_data.txt"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prompt Input Area
    user_prompt = st.text_input(
        "Enter your prompt:", 
        placeholder="e.g., Who is most likely to win a Nobel prize for a mistake?",
    )

    # --- Button Layout ---
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    analyze_clicked = col_btn1.button("🔍 Analyze Now", use_container_width=True)
    surprise_clicked = col_btn2.button("✨ Surprise Me", use_container_width=True)

    selected_prompt = None

    if analyze_clicked:
        if user_prompt:
            selected_prompt = user_prompt
        else:
            st.warning("You need to enter a prompt first! 😅")

    if surprise_clicked:
        if os.path.exists("surprise.txt"):
            with open("surprise.txt", "r") as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            
            if prompts:
                selected_prompt = random.choice(prompts)
                st.info(f"🎲 Random Prompt: **{selected_prompt}**")
            else:
                st.error("The 'surprise.txt' file is empty!")
        else:
            st.error("Missing 'surprise.txt' file!")

    # 4. Execution & Results
    if selected_prompt:
        with st.spinner("🕵️ Scanning chat patterns..."):
            user_corpus = parse_chat(temp_filename)
            
            if not user_corpus.empty:
                results = find_most_likely(selected_prompt, user_corpus)
                
                # Success Celebration!
                st.balloons()
                st.markdown(f"### Results for: *{selected_prompt}*")
                
                # --- Display Results as Cards ---
                for i, (idx, row) in enumerate(results.iterrows()):
                    # Highlight the winner differently
                    border_color = "#FFD700" if i == 0 else "rgba(255, 255, 255, 0.2)"
                    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "👤"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card" style="border: 2px solid {border_color}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 24px;">{icon} <b>{row['Sender']}</b></span>
                                <span style="font-size: 18px; opacity: 0.8;">Rank #{i+1}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_score, col_progress = st.columns([1, 4])
                        with col_score:
                            st.write(f"**{row['Score']*100:.1f}% Match**")
                        with col_progress:
                            score_val = min(max(float(row['Score']), 0.0), 1.0)
                            # Progress bar color changes based on rank
                            bar_color = "rainbow" if i == 0 else "normal"
                            st.progress(score_val)
                        
                        # Detail breakdown in small text
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.caption(f"💎 Score: {row['Score']:.3f}")
                        m_col2.caption(f"🧠 Context: {row['Context_Match']:.3f}")
                        m_col3.caption(f"🎭 Personality: {row['Personality_Match']:.3f}")
                        st.write("") # Padding
            else:
                st.error("Uh oh! We couldn't read that chat format. Make sure it's a standard WhatsApp export.")

    # Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2 style="font-size: 50px;">👋 Welcome!</h2>
        <p style="font-size: 20px;">Upload a WhatsApp .txt file in the sidebar to reveal your group's secrets!</p>
    </div>
    """, unsafe_allow_html=True)