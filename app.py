import streamlit as st
import pandas as pd
import os
from pred import predict_all

st.set_page_config(page_title="WhatPainting? AI", page_icon="🎨", layout="wide")

st.markdown("""
    <style>
    /* Main background color - Soft cream/off-white */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Custom headers matching sketch cards */
    .pastel-header-orange {
        background-color: #FFE6D5;
        color: #4A3B32;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 12px;
    }
    .pastel-header-blue {
        background-color: #E2EFF1;
        color: #2F414F;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Visual borders around cards */
    .metric-card {
        border: 2px solid #E6DCD3;
        border-radius: 12px;
        padding: 16px;
        background-color: #FFFFFF;
        margin-bottom: 20px;
    }
    
    /* Styled Game Banner text */
    .game-title {
        font-family: 'Comic Sans MS', sans-serif; /* Capturing that handwritten vibe */
        font-size: 42px;
        font-weight: 900;
        color: #1A2E40;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }
    .game-subtitle {
        text-align: center;
        font-weight: bold;
        color: #4A5A6A;
        margin-bottom: 24px;
    }
    </style>
""", unsafe_allow_value=True)

PAINTING_IMAGES = {
    "The Starry Night": "starry_night.jpg",
    "The Persistence of Memory": "persistence.jpg",
    "The Water Lily Pond": "water_lilies.jpg"
}


left_panel, right_panel = st.columns([1, 2], gap="large")


with left_panel:
    
    # 1. About The Model
    st.markdown('<div class="pastel-header-orange">ABOUT THE MODEL</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <strong>Measured Accuracy:</strong> 90.53%<br>
        <strong>Framework:</strong> NumPy<br>
        <strong>Constraints:</strong> &lt; 10MB Model Size
    </div>
    """, unsafe_allow_html=True)
    
    # 2. How Does It Work?
    st.markdown('<div class="pastel-header-orange">HOW DOES IT WORK?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card" style="font-size: 14px; line-height: 1.6;">
        This AI was trained on 1,686 subjective survey responses from UofT students. 
        By studying the text frequency (TF-IDF) of your descriptions and the character n-grams 
        of your words, it identifies patterns in how humans perceive these three art pieces.
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Meet The Team
    st.markdown('<div class="pastel-header-orange">MEET THE TEAM</div>', unsafe_allow_html=True)
    st.write("This model originally was a couple of Python files composed to complete a group project for an Introduction to Machine Learning course (CSC311), completed by the following team!")
    
    # Avatars row
    team_col1, team_col2, team_col3 = st.columns(3)
    with team_col1:
        st.markdown("<div style='text-align: center; font-size: 32px;'>👤</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 12px; font-weight: bold;'>Heba Syed</p>", unsafe_allow_html=True)
    with team_col2:
        st.markdown("<div style='text-align: center; font-size: 32px;'>👤</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 12px; font-weight: bold;'>Madiha Khan</p>", unsafe_allow_html=True)
    with team_col3:
        st.markdown("<div style='text-align: center; font-size: 32px;'>👤</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 12px; font-weight: bold;'>Samaah Abdullah</p>", unsafe_allow_html=True)



with right_panel:
    st.markdown('<div class="game-title">GUESS THE PAINTING</div>', unsafe_allow_html=True)
    st.markdown('<div class="game-subtitle">PICK A PAINTING, LET ME GUESS IT!</div>', unsafe_allow_html=True)
    
    art_col1, art_col2, art_col3 = st.columns(3)
    with art_col1:
        st.image("starry_night.jpg", caption="PAINTING A: STARRY NIGHT", use_container_width=True)
    with art_col2:
        st.image("persistence.jpg", caption="PAINTING B: PERSISTENCE OF MEMORY", use_container_width=True)
    with art_col3:
        st.image("water_lilies.jpg", caption="PAINTING C: WATER LILIES", use_container_width=True)
        
    st.info("💡 Pick one of the paintings above in your mind, then fill out the survey below. Let's see if the AI can guess which one you choose!")

    with st.container(border=True):
        vibe = st.text_area("① Describe how this painting makes you feel using adjectives")
        food = st.text_input("② If this painting was a food, what would it be?")
        soundtrack = st.text_area("③ Imagine a soundtrack for this painting. Describe it without naming objects.")
        
        intensity = st.slider("Rate emotional intensity when looking at this painting (1-10, 1 being least, 10 being most)", 1, 10, 5)
        
        submitted = st.form_submit_button("PREDICT PAINTING") if 'form' in locals() else st.button("PREDICT PAINTING", use_container_width=True)

    if submitted:
        if not vibe or not soundtrack:
            st.error("Please fill out the fields to run the model pipeline!")
        else:
            input_data = {
                "Describe how this painting makes you feel.": [vibe],
                "If this painting was a food, what would be?": [food],
                "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": [soundtrack],
                "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": [intensity],
                "How many prominent colours do you notice in this painting?": [0],
                "How many objects caught your eye in the painting?": [0],
                "How much (in Canadian dollars) would you be willing to pay for this painting?": ["0"],
                "This art piece makes me feel sombre.": ["3"],
                "This art piece makes me feel content.": ["3"],
                "This art piece makes me feel calm.": ["3"],
                "This art piece makes me feel uneasy.": ["3"],
                "If you could purchase this painting, which room would you put that painting in?": [""],
                "If you could view this art in person, who would you want to view it with?": [""],
                "What season does this art piece remind you of?": [""]
            }

            temp_filename = "temp_input.csv"
            pd.DataFrame(input_data).to_csv(temp_filename, index=False)

            with st.spinner("AI is processing the text features..."):
                try:
                    prediction = predict_all(temp_filename)[0]
                    
                    st.balloons()
                    st.markdown("---")
                    st.success(f"### 🎉 The AI Guesses: **{prediction}**")
                    
                    if prediction in PAINTING_IMAGES:
                        st.image(PAINTING_IMAGES[prediction], caption=f"Match Found: {prediction}", width=400)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
