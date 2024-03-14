import streamlit as st
from PIL import Image
from fastai.vision.all import load_learner
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io

st.title("Music Genre Classifier")


st.write("Upload your music song here")

with st.spinner('Loading model...'):
    learn_inf = load_learner('music_genre_classifier.pkl')
st.success('Model loaded successfully')

try:
    uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav', 'audio/mpeg'])
    upload_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        data, sr = librosa.load(uploaded_file)
        mel_spectogram = librosa.feature.melspectrogram(y=data, sr=sr)
        mel_spectogram = librosa.power_to_db(mel_spectogram, ref=np.max)
        librosa.display.specshow(mel_spectogram, x_axis='time', y_axis='mel', sr=sr)
        plt.title('Mel Spectogram')
        plt.colorbar(format='%+2.0f dB')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        st.image(im, caption='Mel Spectogram', use_column_width=True)
        st.write("File uploaded successfully")
        st.write("Predicting the genre of the song...")
        pred, pred_idx, probs = learn_inf.predict(mel_spectogram)
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")
    if upload_image is not None:
        image = Image.open(upload_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("File uploaded successfully")
        st.write("Predicting the genre of the song...")
        pred, pred_idx, probs = learn_inf.predict(image)
        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")

except Exception as e:
    print(e)
    st.write("Error when processing the uploaded file")




