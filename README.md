# 🎨 WhatPainting? AI Sentiment Classifier

An interactive, multimodal machine learning application that identifies iconic artworks based entirely on subjective, human sentiment and open-ended descriptions. 

🚀 **[Click here to try the live Web Dashboard!](https://whichpainting.streamlit.app/)**

## 📊 Project Specifications
* **Validation Accuracy:** 90.53%
* **Model Size:** < 10MB (Strict constraint)
* **Dependency Constraint:** Zero external machine learning frameworks utilized during execution (Pure NumPy inference).

## 🧠 Core Engineering Pipeline
1. **Multimodal Feature Fusion:** Merges textual survey responses (TF-IDF word and character n-grams) with structured behavioral elements (Likert scales and log-scaled valuations).
2. **Custom Text Tokenizer:** Normalizes inputs and handles typos dynamically via character-level n-grams.
3. **Optimized Linear Inference:** Executes high-speed matrix multiplications directly via custom weights and biases.

## 👥 The Engineering Team
* **Heba Syed** 
* **Madiha Khan** 
* **Samaah Abdullah** 
