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
    * **Measured Accuracy:** 90.53%
    * **Framework:** Custom NumPy Inference (No Scikit-learn)
    * **Constraints:** Under 10MB Model Size
    """)
    
    st.write("---")
    
    st.header("🧠 HOW DOES IT WORK?")
    st.info("""
    This AI was trained on 1,686 subjective survey responses from UofT students. 
    By studying the text frequency (TF-IDF) of your descriptions and the character n-grams 
    of your words, it identifies patterns in how humans perceive these three art pieces.
    """)
    
    st.write("---")
    
    st.header("👥 MEET THE TEAM")
    st.write("This model originally started as a machine learning project for CSC311. Built by:")
    
    team_col1, team_col2, team_col3 = st.columns(3)
    with team_col1:
        st.subheader("👤")
        st.caption("**Heba Syed**")
    with team_col2:
        st.subheader("👤")
        st.caption("**Madiha Khan**")
    with team_col3:
        st.subheader("👤")
        st.caption("**Samaah Abdullah**")

with right_panel:
    st.title("🎨 GUESS THE PAINTING")
    st.markdown("#### **PICK A PAINTING, LET ME GUESS IT!**")
    
    art_col1, art_col2, art_col3 = st.columns(3)
    with art_col1:
        st.image("starry_night.jpg", caption="PAINTING A: STARRY NIGHT", use_container_width=True)
    with art_col2:
        st.image("persistence.jpg", caption="PAINTING B: PERSISTENCE OF MEMORY", use_container_width=True)
    with art_col3:
        st.image("water_lilies.jpg", caption="PAINTING C: WATER LILIES", use_container_width=True)
        
    st.warning("💡 Pick one of the paintings above in your mind, then fill out the survey below. Let's see if the AI can guess which one you choose!")

    with st.container(border=True):
        vibe = st.text_area("① Describe how this painting makes you feel using adjectives", placeholder="e.g., cold, swirly, melancholic...")
        food = st.text_input("② If this painting was a food, what would it be?", placeholder="e.g., black coffee, melted cheese...")
        soundtrack = st.text_area("③ Imagine a soundtrack for this painting. Describe it without naming objects.", placeholder="e.g., low cello notes with a chaotic tempo...")
        
        intensity = st.slider("Rate emotional intensity when looking at this painting (1-10, 1 being least, 10 being most)", 1, 10, 5)
        
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

            with st.spinner
