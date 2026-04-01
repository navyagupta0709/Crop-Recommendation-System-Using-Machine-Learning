import streamlit as st
import numpy as np
import pickle

# Load models
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Title
st.title("🌾 Crop Recommendation System")

st.write("Enter the required soil and environmental parameters:")

# Input fields
N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

# Predict button
if st.button("Predict Crop"):

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    try:
        # Scaling
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)

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
            st.success(f"🌱 {crop} is the best crop to be cultivated.")
        else:
            st.error("❌ Could not determine crop.")

    except Exception as e:
        st.error(f"Error: {e}")
