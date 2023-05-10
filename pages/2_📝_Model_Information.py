import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle

st.set_page_config(
    page_title="Model Information",
    page_icon="📝",
    layout="wide"
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.markdown("# Machine Learning Model Information")

st.sidebar.header("Model Information")
st.sidebar.markdown("Model Information module presents information and specifications regarding the developed machine learning model.")

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

with open('model_metrics.pkl', 'rb') as g:
    model_metrics = pickle.load(g)

with open('roc_data.pkl', 'rb') as h:
    fpr, tpr, thresholds, roc_auc = pickle.load(h)

st.subheader("Machine Learning Model Overview")

with st.expander("Model Overview",expanded=True):

    st.write("")
    st.write(f"**Model Algorithm :** {model_info['algorithm']}")
    st.write("**Model Input :** Features Exist in Dataset ")
    st.write("**Model Output :** Natural Occurence or Attack Occurence (0/1)")

col1, col2 = st.columns(2)

with col1:
    with st.expander("Model Parameters",expanded=True):
        params = model_info['parameters']
        df_params = pd.DataFrame.from_dict(params, orient='index', columns=['Value'])
        df_params['Value'] = df_params['Value'].astype(str)
        st.write("")
        st.dataframe(df_params, use_container_width=True)

with col2:
    with st.expander("Input Features",expanded=True):
        features = model_info['features']
        df_features = pd.DataFrame({'Features': features})
        st.write("")
        st.dataframe(df_features, use_container_width=True)


st.markdown("***")
st.write("")

st.subheader("Machine Learning Model Metrics")
col3, col4 = st.columns(2)
with col3:
    with st.expander("Metrics",expanded=True):
        st.write("")
        st.write("**Training Set Accuracy** : ", format(model_metrics['train_accuracy'] * 100, ".1f"), "%")
        st.write("**Model Accuracy** :  ", format(model_metrics['accuracy'] * 100, ".1f"), "%")
        st.write("**Model F1-score** : ", format(model_metrics['f1_score'] * 100, ".1f"), "%")
        st.write("**Precision** : ", format(model_metrics['precision'] * 100, ".1f"), "%")
        st.write("**Recall** : ", format(model_metrics['recall']* 100, ".1f"), "%")
        st.write("")
        st.write("**Classification Report**: ")
        report = model_metrics['classification_report']
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report, use_container_width=True)
        st.write("")
        st.write("")
        st.write("")

with col4:
    with st.expander("Confusion Matrix",expanded=True):
        st.write("")
        st.write("")
        cm = model_metrics['confusion_matrix']
        cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        df_confusion_matrix = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
        chart = alt.Chart(df_confusion_matrix.reset_index().melt('index')).mark_rect().encode(
            x=alt.X('variable:O', title='Predicted Label'),
            y=alt.Y('index:O', title='True Label'),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='blues'), title=''),
            size=alt.Size('value:Q', scale=alt.Scale(type='sqrt'), title='Count'),
        ).properties(
            width=400,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        st.info("""The Natural & Attack Occurence are labelled as follows: \n\n  *Natural :   0 , Attack  :   1*""", icon="ℹ️")

    
with st.expander("ROC-AUC Curve",expanded=True):
    st.write("")
    roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': thresholds})

    roc_curve = alt.Chart(roc_df).mark_line().encode(
        x='False Positive Rate',
        y='True Positive Rate'
    )
    diag_line = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='red', strokeDash=[5,2]).encode(
        x='x',
        y='y'
    )
    roc_chart = (roc_curve + diag_line).properties(
        width=500,
        height=500,
        title=f'ROC Curve (AUC={roc_auc:.2f})'
    )
    st.altair_chart(roc_chart, use_container_width = True)
