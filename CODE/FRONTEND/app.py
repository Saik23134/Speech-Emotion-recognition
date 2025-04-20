import streamlit as st
import pickle
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import os

# Login credentials
USERNAME = "Admin"
PASSWORD = "123"

# Add background image using CSS
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://daily.jstor.org/wp-content/uploads/2021/10/how_to_hear_images_and_see_sounds_1050x700.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color:red;
        }
        p{
        color:orange;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# Load label encoder
label_encoder_path = 'label_encoder.pkl'  # Make sure this path is correct
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Load the pre-trained models
lstm_model = load_model('lstm.keras')
cnn_model = load_model('cnn_model.keras')

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).tolist()

# Function to make predictions
def predict_audio(model, file_path):
    features = extract_features(file_path)
    features = np.array(features).reshape(1, -1)  # Reshaping to match model input
    prediction = model.predict(features)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Streamlit login page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()

        else:
            st.error("Incorrect username or password")

# Main Streamlit app
def main():
    # Set background image
    set_background_image()

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    else:
        # Sidebar with radio buttons for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home Page", "Prediction Page"])

        if page == "Home Page":
            st.title("Home Page")
            st.write("""
                # Speech Emotion Classification
                This project involves classifying emotions in speech. We use a combination of 
                machine learning models (LSTM and CNN) to predict the emotion of speech based 
                on audio files. The model has been trained on various speech emotion datasets.
            """)

        elif page == "Prediction Page":
            st.title("Prediction Page")
            st.write("Upload an audio file and select a model to make a prediction.")

            # Upload audio file
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Select the model for prediction
                model_choice = st.radio("Choose the model", ("LSTM", "CNN"))

                if st.button("Predict"):
                    # Make prediction based on the selected model
                    if model_choice == "LSTM":
                        prediction = predict_audio(lstm_model, "temp_audio.wav")
                    else:
                        prediction = predict_audio(lstm_model, "temp_audio.wav")
                    
                    st.success(f"The predicted emotion is: {prediction.capitalize()}")

if __name__ == "__main__":
    main()
