import streamlit as st
import pandas as pd
import os
from pred import predict_all

st.set_page_config(page_title="WhatPainting? AI Classifier", page_icon="🎨", layout="wide")

PAINTING_IMAGES = {
    "The Starry Night": "starry_night.jpg",
    "The Persistence of Memory": "persistence.jpg",
    "The Water Lily Pond": "water_lilies.jpg"
}


left_panel, right_panel = st.columns([1, 2], gap="large")


with left_panel:
    
    st.header("💡 ABOUT THE MODEL")
    st.markdown("""
    <div style="font-size: 18px; line-height: 1.6;">
    <ul>
        <li><strong>Measured Accuracy:</strong> 90.53%</li>
        <li><strong>Framework:</strong> Custom NumPy Inference (No Scikit-learn)</li>
        <li><strong>Constraints:</strong> Under 10MB Model Size</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    st.header("🧠 HOW DOES IT WORK?")
    st.markdown("""
    <div style="font-size: 18px; line-height: 1.6; background-color: rgba(28, 131, 225, 0.1); padding: 15px; border-radius: 10px;">
        This AI was trained on 1,686 subjective survey responses from UofT students. 
        By studying the text frequency (TF-IDF) of your descriptions and the character n-grams 
        of your words, it identifies patterns in how humans perceive these three art pieces.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    st.header("👥 MEET THE TEAM")
    st.markdown("""
    <div style="font-size: 18px; line-height: 1.6; margin-bottom: 15px;">
        This model originally started as a machine learning project for CSC311. Built by:
    </div>
    """, unsafe_allow_html=True)
    
    team_col1, team_col2, team_col3 = st.columns(3)
    with team_col1:
        st.image("heba.png", use_container_width=True)
        st.markdown("<p style='text-align: center; font-size: 16px; font-weight: bold;'>Heba Syed</p>", unsafe_allow_html=True)
    with team_col2:
        st.image("madeeha.png", use_container_width=True)
        st.markdown("<p style='text-align: center; font-size: 16px; font-weight: bold;'>Madeeha Khan</p>", unsafe_allow_html=True)
    with team_col3:
        st.image("samaah.png", use_container_width=True)
        st.markdown("<p style='text-align: center; font-size: 16px; font-weight: bold;'>Samaah Abdullah</p>", unsafe_allow_html=True)



with right_panel:
    st.title("🎨 GUESS THE PAINTING")
    st.markdown("## **PICK A PAINTING, LET ME GUESS IT!**")
    
    art_col1, art_col2, art_col3 = st.columns(3)
    with art_col1:
        st.image("starry_night.jpg", caption="PAINTING A: STARRY NIGHT", use_container_width=True)
    with art_col2:
        st.image("persistence.jpg", caption="PAINTING B: PERSISTENCE OF MEMORY", use_container_width=True)
    with art_col3:
        st.image("water_lilies.jpg", caption="PAINTING C: WATER LILIES", use_container_width=True)
        
    st.markdown("""
    <div style="font-size: 18px; background-color: rgba(255, 165, 0, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        💡 <strong>Pick one of the paintings above in your mind</strong>, then fill out the survey below. Let's see if the AI can guess which one you choose!
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: -10px;'><span style='color: #FFB347;'>1.</span> Describe how this painting makes you feel using adjectives</p>", unsafe_allow_html=True)
        vibe = st.text_area("", placeholder="e.g., cold, swirly, melancholic...", key="vibe_input")
        
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: -10px;'><span style='color: #FFB347;'>2.</span> If this painting was a food, what would it be?</p>", unsafe_allow_html=True)
        food = st.text_input("", placeholder="e.g., black coffee, melted cheese...", key="food_input")
        
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: -10px;'><span style='color: #FFB347;'>3.</span> Imagine a soundtrack for this painting. Describe it without naming objects.</p>", unsafe_allow_html=True)
        soundtrack = st.text_area("", placeholder="e.g., low cello notes with a chaotic tempo...", key="soundtrack_input")
        
        st.markdown("<p style='font-size: 18px; font-weight: bold; margin-bottom: 5px;'>Rate emotional intensity when looking at this painting (1-10, 1 being least, 10 being most)</p>", unsafe_allow_html=True)
        intensity = st.slider("", 1, 10, 5, key="intensity_slider")
        
        st.write("")
        submitted = st.button("PREDICT PAINTING", use_container_width=True)

    if submitted:
        if not vibe or not soundtrack:
            st.error("Please fill out the text fields to generate features for the pipeline!")
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

            with st.spinner("Running Matrix Multiplications..."):
                try:
                    prediction = predict_all(temp_filename)[0]
                    
                    st.balloons()
                    st.write("---")
                    st.markdown(f"<h2 style='color: #2ea44f;'>🎉 The AI Guesses: {prediction}</h2>", unsafe_allow_html=True)
                    
                    if prediction in PAINTING_IMAGES:
                        st.image(PAINTING_IMAGES[prediction], caption=f"Match Found: {prediction}", width=500)
                except Exception as e:
                    st.error(f"Inference pipeline crash: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
