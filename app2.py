import pickle
import numpy as np
import streamlit as st

# Load the trained model
# Deserialization where you turn the byte stream back to the original python object
with open("diamond_price_model (1).pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.markdown("<h1 style='text-align: center;'>💎 Diamond Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Enter diamond features to estimate price.</h6>", unsafe_allow_html=True)

# Input layout
col1, col2 = st.columns(2)

with col1: #Column 1
    carat = st.number_input("Carat", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    depth = st.number_input("Depth %", min_value=50.0, max_value=70.0, value=61.0)
    x = st.number_input("X (length in mm)", min_value=0.0, max_value=10.0, value=4.0)
    z = st.number_input("Z (depth in mm)", min_value=0.0, max_value=10.0, value=2.5)

with col2: #column 2
    color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    table = st.number_input("Table %", min_value=50.0, max_value=70.0, value=57.0)
    y = st.number_input("Y (width in mm)", min_value=0.0, max_value=10.0, value=4.0)

# Encode categorical features
cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

# Input Vector (In Exact Order of how the RF model was trained) 
input_data = np.array([[carat,
                        cut_map[cut],
                        color_map[color],
                        clarity_map[clarity],
                        depth,
                        table,
                        x,
                        y,
                        z]])

# Predict button
if st.button("Predict Price 💰"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Diamond Price: ${prediction[0]:,.2f}")
