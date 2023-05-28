import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from PIL import Image
import time

st.set_page_config(
    page_title="Data Visualization",
    page_icon="📈",
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.markdown("# Data Visualization")
st.subheader("Dataset")

st.sidebar.header("Data Visualization")
st.sidebar.markdown("Data Visualization module presents statistics and charts in relation to power system attack dataset.")


def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        return None
    
with open('features_list.pkl', 'rb') as file:
    required_features = pickle.load(file)

with st.expander("Upload Dataset",expanded=True):
    st.write("")

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload Dataset - CSV File", type="csv", help="Click Browse Files to upload the dataset in CSV file format.")
        submit_button = st.form_submit_button(label="Upload")

        if uploaded_file is not None and submit_button:
            data = load_data(uploaded_file)
            
            if data is not None and set(required_features).issubset(data.columns):
                st.session_state.data = data
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_size = uploaded_file.size
                st.success("Dataset File successfully uploaded.")
            else:
                st.error("The uploaded CSV file does not have the required features.")
    
    st.write("")


reset_session = st.button("Clear Dataset")

if reset_session:
    st.session_state.data = None
    st.session_state.file_name = None
    st.session_state.file_size = None
    uploaded_file = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'file_name' not in st.session_state:
    st.session_state.file_name = None

if 'file_size' not in st.session_state:
    st.session_state.file_size = None

data = st.session_state.data
file_name = st.session_state.file_name

if st.session_state.file_size is not None:
    file_size = st.session_state.file_size / (1024 * 1024)

def get_feature_count(data):
    features = data.drop(columns=['marker']).columns.tolist()
    num_features = len(features)
    st.write(f"Total Number of Features Excluding Target Feature: ", num_features)
    return

def display_features(data):
    st.write("List of All Features:")
    st.caption("*Expand to view list of Features.*")
    st.write(data.columns.tolist())

def target_variable(data):
    for column in data.select_dtypes(include=['object']):
        st.write("Target Feature for the Dataset: ", column)
        st.write('\nUnique Values for', column, ':')
        st.dataframe(data[column].value_counts(), use_container_width=True)

def split_data(data):
    new_data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = new_data.drop("marker", axis=1)
    y = new_data["marker"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X, y, X_train, X_test, y_train, y_test

def feature_importance(X_train, y_train, X): 
    f_scores, p_values = f_classif(X_train, y_train)
    features_df = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores})
    features_df = features_df.sort_values('F-Score', ascending=False)

    return features_df

def explore_data(data):

    st.markdown("***")
    st.subheader("Dataset Information & Statistics")
    
    with st.expander("Display Dataset Head"):
        st.write("")
        st.write(data.head())

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Basic Dataset Information",expanded=True):
            st.write("")
            st.write(f"Total Number of Data: ", data.shape[0])

            num_variables = len(data.columns)
            st.write(f"Total Number of Features: ", num_variables)

            get_feature_count(data)

        with st.expander("Feature Data Types", expanded=True):
            data_types = data.dtypes.value_counts().rename_axis('data_type').reset_index(name='count')
            data_types.columns = ["data_type", "count"]
            data_types['data_type'] = data_types['data_type'].replace({'int64': 'Integer', 'float64': 'Float', 'object': 'Categorical'})

            st.write("")
            st.write("")
            st.info("""Data types of features are the different ways that data can be represented. 
                These categories can be used to describe the type of data, such as categorical or numerical.""", icon="ℹ️")
            
            st.write("")
            st.write("")
            st.dataframe(data_types, use_container_width=True)
            st.write("")
            st.write("")
            chart = alt.Chart(data_types.reset_index()).mark_bar().encode(x=alt.X('data_type', title='Data Type'), y=alt.Y('count', title='Number of Features'))
            st.altair_chart(chart, use_container_width=True)

        with st.expander("Features",expanded=True):
            st.write("")
            display_features(data)
            st.markdown("***")

            st.write("**Features Explanation:**")
            feature_table = Image.open('feature_table.png')
            st.image(feature_table, caption="Features Explanation")

            st.markdown("***")
            
            st.write("**Target Feature:**")
            st.info("""The target feature is the variable that user want to predict or explain in a dataset. 
                It is the outcome or response variable of interest in a machine learning or statistical analysis task.""", icon="ℹ️")

            st.write("")
            st.write("")
            target_variable(data)
            
            target_feature = 'marker'
            target_data = pd.DataFrame(data[target_feature].value_counts()).reset_index()
            target_data.columns = [target_feature, 'count']

            chart_2 = alt.Chart(target_data).mark_bar().encode(x=target_feature, y='count')

            st.altair_chart(chart_2, use_container_width=True)

    with col2:
        with st.expander("Power System Framework", expanded=True):
            framework = Image.open('ps_framework.JPG')
            st.write("")
            st.image(framework, caption="Power System Framework Configuration")
            st.write("")
            st.write("""
            Configuration Details
            - Power Generators: G1, G2
            - Intelligent Electronic Devices (Relay): R1, R2, R3, R4
            - Breakers: BR1, BR2, BR3, BR4 """)
            
            st.write("")
            st.write("")
            st.info("""The figure shows the power system framework configuration used in generating the power system scenarios 
            performed by Mississippi State University & Oak Ridge National Laboratory.""", icon="ℹ️")
        
        with st.expander("Missing Values & Infinity Values", expanded=True):
            st.write("")
            st.info("""Missing values are data points that are not present in the dataset. 
            Infinity values are data points that are larger or smaller than any possible value.""", icon="ℹ️")
            st.write("")
            
            total_missing = data.isnull().sum().sum()
            st.write(f"Total Number of Missing Values: ", total_missing)

            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            infinite_values = np.isinf(data[numeric_cols]).sum().sum()

            st.write("Total Number of Infinity Values: ", infinite_values)

            inf_cols = data.replace([np.inf, -np.inf], np.nan).isna().sum()[data.dtypes != 'object'].sort_values(ascending=False)
            inf_cols = inf_cols[inf_cols > 0]

            if not inf_cols.empty:
                st.write("")
                st.write("")
                inf_df = pd.DataFrame()
                inf_df['Feature'] = inf_cols.index
                inf_df['Infinity Values'] = inf_cols.values
                
                st.write("Features with Infinity Values:")
                chart_3 = alt.Chart(inf_df).mark_bar().encode(
                x='Feature',
                y='Infinity Values',
                tooltip=['Feature', 'Infinity Values']
                )
                st.altair_chart(chart_3, use_container_width=True)
            else:
                st.write("No columns with infinity values found.")

        with st.expander("Feature Importance", expanded=True):
            st.write("")
            st.info("""Feature importance is the process used in assigning scores to features in the dataset to indicate how important they are for predicting the target variable. 
            This method is used to select the most important features for the machine learning model.""", icon="ℹ️")
            st.write("")
            st.write("Features Ranked based on ANOVA F-value.")

            st.info("""ANOVA F-value feature selection method is used  as a statistical method which uses the F-statistic to select 
            features that are most likely to be important for the machine learning model.""", icon="ℹ️")
            st.write("")

            X, y, X_train, X_test, y_train, y_test = split_data(data)
            
            features_df = feature_importance(X_train, y_train, X)

            num_features = st.selectbox("Chart Display Options", options=["Top 10 Features", "All Features", "All Features Ranked"], index=0)

            st.write("")
            st.write("")

            if num_features == "All Features":
                chart_4 = alt.Chart(features_df).mark_bar().encode(
                    x=alt.X('F-Score:Q', title='ANOVA F-Score'),
                    y=alt.Y('Feature:N', title='Features'),
                    tooltip=[alt.Tooltip('Feature:N'), alt.Tooltip('F-Score:Q', format='.2f')] 
                ).properties(height=1200)

                st.altair_chart(chart_4, use_container_width=True)
            
            elif num_features == "All Features Ranked":
                chart_4 = alt.Chart(features_df).mark_bar().encode(
                    x=alt.X('F-Score:Q', title='ANOVA F-Score'),
                    y=alt.Y('Feature:N', title='Features', sort=alt.EncodingSortField('F-Score', order='descending')),
                    tooltip=[alt.Tooltip('Feature:N'), alt.Tooltip('F-Score:Q', format='.2f')] 
                ).properties(height=1200)

                st.altair_chart(chart_4, use_container_width=True)

            else:
                top_features = features_df[:10]
                chart_4 = alt.Chart(top_features).mark_bar().encode(
                    x=alt.X('F-Score:Q', title='ANOVA F-Score'),
                    y=alt.Y('Feature:N', title='Features', sort=alt.EncodingSortField('F-Score', order='descending')),
                    tooltip=[alt.Tooltip('Feature:N'), alt.Tooltip('F-Score:Q', format='.2f')] 
                )

                st.altair_chart(chart_4, use_container_width=True)

if data is not None:
    st.info(f"""**Current Loaded Dataset:** \n\n File Name: {file_name} \n\n File Size: {file_size:.2f} MB""", icon="ℹ️")
    
    explore_data(data)
