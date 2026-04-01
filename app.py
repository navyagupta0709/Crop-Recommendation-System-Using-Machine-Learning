import streamlit as st
import numpy as np
import pickle

# Load model & scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>🌱 Crop Recommendation</h1>", unsafe_allow_html=True)

# Layout (2 columns like your image)
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", placeholder="Enter Nitrogen")
    temp = st.number_input("Temperature (°C)")
    rainfall = st.number_input("Rainfall (mm)")

with col2:
    P = st.number_input("Phosphorus")
    K = st.number_input("Potassium")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("pH")

# Button centered
st.markdown("<br>", unsafe_allow_html=True)
center = st.columns([1,2,1])

with center[1]:
    predict_btn = st.button("Get Recommendation")

# Prediction
if predict_btn:
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    try:
        scaled = ms.transform(single_pred)
        final = sc.transform(scaled)
        prediction = model.predict(final)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            st.success(f"🌾 {crop} is best crop to cultivate")
            st.balloons()
        else:
            st.error("❌ Could not determine crop")

    except Exception as e:
        st.error(f"Error: {e}")

# Image at bottom
st.markdown("---")
st.image("crop.jpg", use_column_width=True)
