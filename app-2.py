import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model

st.title('Facial Emotion Recognition')

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


@st.cache_resource
def get_model():
    return load_model('emotion_model.keras', compile=False)


model = get_model()

file = st.camera_input('Take a photo (face in the center)')


if file is not None:
    img = Image.open(file).convert('L')                 # grayscale like the training data
    small = ImageOps.fit(img, (48, 48))                  # center crop + resize to 48x48

    probs = model.predict(np.array(small).reshape(1, 48, 48), verbose=0)[0]

    st.subheader('Prediction: ' + class_names[probs.argmax()])
    st.bar_chart(pd.Series(probs, index=class_names))
    st.image(small, caption='what the model sees (48x48)', width=150)