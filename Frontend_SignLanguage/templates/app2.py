import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

st.title("Multi Sign Language Speech and Text Converter")

# Load models
model_asl_dict = pickle.load(open('model_ASL.pkl','rb'))
model_asl = model_asl_dict['model']

model_isl_dict = pickle.load(open('model_ISL.pkl','rb'))
model_isl = model_isl_dict['model']

model_ssl_dict = pickle.load(open('model_SSL.pkl','rb'))
model_ssl = model_ssl_dict['model']

model_bsl_dict = pickle.load(open('model_BSL.pkl','rb'))
model_bsl = model_bsl_dict['model']

# Select language
language = st.selectbox(
    "Choose Sign Language",
    ["ASL","ISL","SSL","BSL"]
)

if language == "ASL":
    model = model_asl
elif language == "ISL":
    model = model_isl
elif language == "SSL":
    model = model_ssl
else:
    model = model_bsl


# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

labels_dict = {chr(i+65): chr(i+65) for i in range(26)}

# Webcam input
image = st.camera_input("Take a picture of your sign")

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv.imdecode(file_bytes, 1)

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

        if len(data_aux) == 42:
            data_aux.extend([0]*42)
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[str(prediction[0])]

        st.subheader("Predicted Character:")
        st.success(predicted_character)