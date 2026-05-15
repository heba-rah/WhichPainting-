import streamlit as st
import pandas as pd
import os
from pred import predict_all

# Mapping result labels to local image files
PAINTING_IMAGES = {
    "The Starry Night": "starry_night.jpg",
    "The Persistence of Memory": "persistence.jpg",
    "The Water Lily Pond": "water_lilies.jpg"
}

st.set_page_config(page_title="WhatPainting? AI", page_icon="🎨")

st.title("🎨 WhatPainting? AI Classifier")
st.markdown("### Describe a painting's 'vibe' to see if the AI can identify it.")

st.sidebar.header("Model Technicals")
st.sidebar.write("**Accuracy:** 90.53%")
st.sidebar.write("**Framework:** Custom NumPy Inference (No Scikit-learn)")
st.sidebar.write("**Constraints:** < 10MB Model Size")

with st.form("survey_form"):
    st.subheader("The Subjective Survey")
    
    vibe = st.text_area("Describe how this painting makes you feel.")
    food = st.text_input("If this painting was a food, what would it be?")
    soundtrack = st.text_area("Imagine a soundtrack for this painting. Describe it without naming objects.")
    
    intensity = st.slider("Emotion intensity (1-10)", 1, 10, 5)
    
    submitted = st.form_submit_button("Predict Painting")

if submitted:
    if not vibe or not soundtrack:
        st.warning("Please provide at least the feeling and soundtrack descriptions!")
    else:
        # Create a dictionary that matches your pred.py's expected columns
        input_data = {
            "Describe how this painting makes you feel.": [vibe],
            "If this painting was a food, what would be?": [food],
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": [soundtrack],
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": [intensity],
            # Fill remaining required columns from pred.py with default values
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

        with st.spinner("Analyzing sentiment..."):
            try:
                prediction = predict_all(temp_filename)[0]
                
                st.balloons()
                st.header(f"Result: {prediction}")
                
                if prediction in PAINTING_IMAGES:
                    st.image(PAINTING_IMAGES[prediction], caption=f"Predicted: {prediction}")
                else:
                    st.write("Image not found, but prediction successful!")
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
