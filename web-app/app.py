import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Constants ---
MAX_LEN = 40

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

# --- Load Assets ---
# We use @st.cache_resource so Streamlit doesn't reload the heavy model 
# every time you click a button or type a letter.
bilstm_sentiment_model="web-app/models/bilstm_sentiment_model.keras"
tokenizer_path="web-app/models/tokenizer.pkl"
label_encoder="web-app/models/label_encoder.pkl"

@st.cache_resource
def load_assets():
    model = load_model(bilstm_sentiment_model)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(label_encoder, "rb") as f:
        le = pickle.load(f)
        
    return model, tokenizer, le

# Show a loading spinner while the model boots up
with st.spinner("Loading AI model..."):
    model, tokenizer, le = load_assets()

# --- Helper Function ---
# (Ensure this matches the cleaning logic you used during training!)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# --- App UI ---
st.title("🧠 Sentiment Analysis AI")
st.write("Enter a sentence below, and the BiLSTM neural network will predict its sentiment.")

# Text input
user_input = st.text_area("Your Text:", height=150, placeholder="e.g., This game is absolutely terrible, waste of money")

# Predict button
if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Preprocess the input
        cleaned_text = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        
        # 2. Make prediction
        preds_prob = model.predict(padded)
        pred_int = np.argmax(preds_prob, axis=1)[0]
        confidence = np.max(preds_prob, axis=1)[0]
        
        # 3. Decode the label
        predicted_label = le.inverse_transform([pred_int])[0]
        
        # 4. Display the results
        st.divider()
        st.subheader("Results")
        
        # Format the output beautifully using columns
        col1, col2 = st.columns(2)
        
        with col1:
            if predicted_label.lower() == "positive":
                st.success(f"**{predicted_label.upper()}**")
            elif predicted_label.lower() == "negative":
                st.error(f"**{predicted_label.upper()}**")
            else:
                st.info(f"**{predicted_label.upper()}**")
                
        with col2:
            st.metric(label="Confidence Score", value=f"{confidence * 100:.1f}%")
            
        # Optional: Show all probabilities
        with st.expander("View raw probabilities"):
            for label_class, prob in zip(le.classes_, preds_prob[0]):
                st.write(f"- **{label_class}**: {prob * 100:.2f}%")