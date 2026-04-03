import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except:
    st.error("❌ Model or scaler files not found!")
    st.stop()

st.set_page_config(page_title="Crop Recommendation", layout="wide")

st.markdown("<h1 style='text-align: center; color: green;'>🌱 Crop Recommendation</h1>", unsafe_allow_html=True)

# Input layout
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", value=90.0)
    temp = st.number_input("Temperature (°C)", value=28.0)
    rainfall = st.number_input("Rainfall (mm)", value=150.0)

with col2:
    P = st.number_input("Phosphorus", value=45.0)
    K = st.number_input("Potassium", value=45.0)
    humidity = st.number_input("Humidity (%)", value=80.0)
    ph = st.number_input("pH", value=6.5)

# Button
c1, c2, c3 = st.columns([1,2,1])
with c2:
    predict_btn = st.button("Get Recommendation")

# Crop mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

if predict_btn:
    try:
        features = np.array([
            float(N), float(P), float(K),
            float(temp), float(humidity),
            float(ph), float(rainfall)
        ]).reshape(1, -1)

        # Scaling
        scaled = ms.transform(features)
        final = sc.transform(scaled)

        # -------- MAIN PREDICTION --------
        prediction = model.predict(final)
        pred_crop = crop_dict[int(prediction[0])]

        st.success(f"🌾 Best Crop: {pred_crop}")

        # -------- PROBABILITY (TOP 5 CROPS) --------
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(final)[0]

            # Sort top 5
            top_indices = np.argsort(probs)[::-1][:5]

            crops = []
            values = []

            for i in top_indices:
                crop_name = crop_dict.get(i+1, f"Crop {i+1}")
                crops.append(crop_name)
                values.append(round(probs[i]*100, 2))

            df = pd.DataFrame({
                "Crop": crops,
                "Confidence (%)": values
            })

            st.markdown("### 📊 Top 5 Crop Recommendations")
            st.dataframe(df)

            # Bar chart
            st.bar_chart(df.set_index("Crop"))

        else:
            st.warning("⚠️ Model does not support probability prediction")

    except Exception as e:
        st.error(f"❌ Error: {e}")
