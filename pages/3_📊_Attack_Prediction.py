import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Attack Prediction",
    page_icon="📊",
    layout="wide"
)

#sidebar
st.sidebar.header("Attack Prediction")
st.sidebar.markdown("Attack Prediction module allows users to enter feature inputs based on the dataset and generate predictions of  occurrence of power system attack.")

@st.cache_data
def load_model():
    with open('new_t10_model.pkl', 'rb') as f:
        logreg_model = pickle.load(f)

    with open('feature_range.pkl', 'rb') as f:
        feature_range = pickle.load(f)
    
    return logreg_model, feature_range


st.markdown("# Power System Attack Prediction")
st.subheader("""Features Input""")
st.caption("Use the Slider Widget to Input Values for each Feature.")
st.markdown("***")

logreg_model, feature_range = load_model()

feature_names = list(feature_range.keys())
num_features = len(feature_names)
mid_point = num_features // 2
first_half = feature_names[:mid_point]
second_half = feature_names[mid_point:]


inputs = []  
col1, col2 = st.columns(2,  gap="large")
for feature in first_half:
    min_val, max_val = feature_range[feature]
    value = col1.slider(f'**{feature}:**', min_value=min_val, max_value=max_val)
    inputs.append(value)
    
for feature in second_half:
    min_val, max_val = feature_range[feature]
    value = col2.slider(f'**{feature}:**', min_value=min_val, max_value=max_val)
    inputs.append(value)

new_input = np.array(inputs).reshape(1, -1)
st.markdown("***")

if st.button("**Classify**"):
    prediction = logreg_model.predict(new_input)[0]
    if prediction == 0:
        st.success('The event is Natural')
    else:
        st.error('The event is Attack')
