
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display

# Load the pre-trained model
model = load_model('model.h5')

# Map model output to emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

# Define emoji mapping for emotions
emotion_emoji_map = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'ps': '😲',  # Pleasant Surprise
    'sad': '😢'
}

# App Title and Header
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload an audio file to analyze the emotion conveyed in the speech.")

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload an audio file in **WAV format**.
2. Wait for the app to process the file.
3. View the predicted emotion
""")

# File uploader for audio input
uploaded_file = st.file_uploader("Upload an audio file (WAV format preferred)", type=['wav'])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("**Uploaded File:** Processing your audio...")

    # Process the audio file
    try:
        # Load audio using librosa
        audio, sr = librosa.load(uploaded_file, duration=3, offset=0.5)

        # Extract MFCCs for prediction
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        # Reshape input for the model
        model_input = np.expand_dims(mfccs_processed, axis=0)

        # Predict emotion
        prediction = model.predict(model_input)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        prediction_confidence = np.max(prediction)

        # Scale the prediction confidence to 0–100
        prediction_confidence_percentage = int(prediction_confidence * 100)

        # Display prediction results
        st.subheader("Predicted Emotion 🎭")
        st.markdown(f"### **{predicted_emotion.capitalize()}**")
        st.progress(prediction_confidence_percentage)

        # Display large emoji
        if predicted_emotion in emotion_emoji_map:
            st.markdown(
                f"<div style='text-align: center; font-size: 100px;'>{emotion_emoji_map[predicted_emotion]}</div>", 
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error processing audio file: {e}")


