import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
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
    st.write(f"**Model Algorithm :** {model_info['algorithm']} Algorithm")
    st.caption("*Model Algorithm is the set of instructions used that take dataset as input and produce a model.*")
    st.write("")
    st.write("**Model Input :** Features Exist in Dataset ")
    st.caption("*Model Input is the data that is fed into the model to train and make predictions.*")
    st.write("")
    st.write("**Model Output :** Natural Occurence or Attack Occurence (0/1)")
    st.caption("*Model Output is the prediction made by the developed machine learning model based on input data.*")
    st.write("")

col1, col2 = st.columns(2)

with col1:
    with st.expander("Model Parameters",expanded=True):
        params = model_info['parameters']
        df_params = pd.DataFrame.from_dict(params, orient='index', columns=['Value'])
        df_params['Value'] = df_params['Value'].astype(str)
        st.write("")
        st.dataframe(df_params, use_container_width=True)

        param_explain = [
            ['C', 'Controls the inverse of regularization strength, where smaller values indicate stronger regularization.', 'Positive float values.'],
            ['class_weight', 'Weights associated with classes to address class imbalance issues during model training.', "'balanced', dictionary, array-like of shape (n_classes,), None, or 'none'."],
            ['dual', 'Dual or primal formulation of logistic regression, with dual being suitable when the number of features is greater than the number of samples.', 'True or False.'],
            ['fit_intercept', 'Specifies if an intercept term should be included in the model.', 'True or False.'],
            ['intercept_scaling', 'Scaling factor for intercept if fit_intercept is set to True.', 'Positive float values.'],
            ['l1_ratio', 'The mixing parameter for elastic net regularization, with 0 representing L2 penalty and 1 representing L1 penalty.', 'Float values between 0 and 1, or None.'],
            ['max_iter', 'Maximum number of iterations for the solver to converge.', 'Positive integer values.'],
            ['multi_class', 'Strategy for handling multiple classes: "ovr" (one-vs-rest) or "multinomial".', "'ovr', 'multinomial', 'auto', or None."],
            ['n_jobs', 'Number of parallel jobs to run for the solver. -1 means using all processors.', 'Integer values or -1, None.'],
            ['penalty', 'Type of regularization applied: "l1", "l2", "elasticnet", or "none".', "'l1', 'l2', 'elasticnet', 'none'."],
            ['random_state', 'Seed used by the random number generator for random weight initialization.', 'Integer values or None.'],
            ['solver', 'Solver algorithm to use: "newton-cg", "lbfgs", "liblinear", "sag", "saga".', "'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'."],
            ['tol', 'Tolerance for stopping criteria.', 'Positive float values.'],
            ['verbose', 'Verbosity mode.', '0, 1, or positive integer values.'],
            ['warm_start', 'Reuse the solution of the previous call to fit as initialization.', 'True or False.']
        ]

        df_param_explain = pd.DataFrame(param_explain, columns=['Parameter', 'Explanation', 'Possible Values'])
        st.write("")

        display_exp = st.checkbox('Display Model Parameters Explanation')
        if display_exp:
            st.table(df_param_explain)




with col2:
    with st.expander("Input Features",expanded=True):
        features = model_info['features']
        df_features = pd.DataFrame({'Features': features})
        st.write("")
        st.dataframe(df_features, use_container_width=True)
        st.write("")
        st.info("""Input Features are the selected features from the dataset that is fed into the model to train and make predictions.""", icon="ℹ️")


st.markdown("***")
st.write("")

st.subheader("Machine Learning Model Metrics")
col3, col4 = st.columns(2)
with col3:
    with st.expander("Metrics",expanded=True):
        st.write("")
        st.write("**Training Set Accuracy** : ", format(model_metrics['train_accuracy'] * 100, ".1f"), "%")
        st.caption("*The proportion of correctly classified instances in the training set.*")
        st.write("")
        st.write("**Model Accuracy** :  ", format(model_metrics['accuracy'] * 100, ".1f"), "%")
        st.caption("*The proportion of correctly classified instances in the test set.*")
        st.write("")
        st.write("**Model F1-score** : ", format(model_metrics['f1_score'] * 100, ".1f"), "%")
        st.caption("*The harmonic mean of precision and recall.*")
        st.write("")
        st.write("**Precision** : ", format(model_metrics['precision'] * 100, ".1f"), "%")
        st.caption("*The proportion of predicted positive instances that are actually positive.*")
        st.write("")
        st.write("**Recall** : ", format(model_metrics['recall']* 100, ".1f"), "%")
        st.caption("*The proportion of actual positive instances that are predicted positive.*")
        st.write("")
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
        st.info("""**Reading Confusion Matrix** \n\n The Natural & Attack Occurence are labelled as follows: \n\n  *Natural :   0 , Attack  :   1* 
        \n The Confusion Matrix Diagram is divided into four sections: 
        \n - Top Left Quadrant: True Positives (Number of Natural events labelled as Natural events)
        \n - Top Right Quadrant: False Positives (Number of Natural events labelled as Attack events)
        \n - Bottom Left Quadrant: False Negatives (Number of Attack events labelled as Natural events)
        \n - Bottom Right Quadrant: True Negatives (Number of Attack events labelled as Attack events)
        """, icon="ℹ️")

    
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