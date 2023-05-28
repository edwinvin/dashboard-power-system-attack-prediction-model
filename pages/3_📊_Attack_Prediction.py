import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Attack Prediction",
    page_icon="📊",
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

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
st.write("*Utilize the Slider Widget to input values for each selected feature.*")
st.write("")
with st.expander("Details on Selected Input Features",expanded=False):
    st.write("")
    st.write("""The top 10 features were selected from a total of 128 features in the dataset using the ANOVA F-value feature selection method. 
    These features were then used as input for the final model, which was trained on the dataset.""")
    st.write("")

    feature_details = {
    'Feature': ['R1-PM8:V', 'R1-PM9:V', 'R1-PM11:I', 'R1-PM12:I', 'R2-PM8:V', 'R2-PM11:I', 'R2-PM12:I', 'R3-PM8:V', 'R3-PM11:I', 'R4-PM11:I'],
    'Description': ['Negative Sequence Voltage Phase Magnitude measured by PMU R1',
                    'Zero Sequence Voltage Phase Magnitude measured by PMU R1',
                    'Negative Sequence Current Phase Magnitude measured by PMU R1',
                    'Zero Sequence Current Phase Magnitude measured by PMU R1',
                    'Negative Sequence Voltage Phase Magnitude measured by PMU R2',
                    'Negative Sequence Current Phase Magnitude measured by PMU R2',
                    'Zero Sequence Current Phase Magnitude measured by PMU R2',
                    'Negative Sequence Voltage Phase Magnitude measured by PMU R3',
                    'Negative Sequence Current Phase Magnitude measured by PMU R3',
                    'Negative Sequence Current Phase Magnitude measured by PMU R4']
    }

    feature_details_df = pd.DataFrame(feature_details)
    st.dataframe(feature_details_df, use_container_width=True)

    st.info("""The table summarizes the selected features, indicating the description for each feature with type of measurement (voltage/current) and 
    the phase magnitude being captured by each PMU (R1, R2, R3, R4). The features include information about positive, negative, and zero voltage/current phase magnitudes.""", icon="ℹ️")

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
        st.success('The Event is predicted to be Natural Event.')
    else:
        st.error('The Event is predicted to be Attack Event.')
