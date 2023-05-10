import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
from sklearn.feature_selection import f_classif
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

st.sidebar.header("Data Visualization")
st.sidebar.markdown("Data Visualization module presents statistics and charts in relation to power system attack dataset.")

@st.cache_data
def load_data():
    data = pd.read_csv('raw_merged_df.csv')
    X_train = pd.read_pickle('X_train.pkl')
    y_train = pd.read_pickle('y_train.pkl')
    
    progress_bar = st.progress(0)
    for i in range(1, 101):
        progress_bar.progress(i)
        time.sleep(0.1)
    
    progress_bar.empty()
    return data, X_train, y_train

def get_feature_count(data):
    features = data.drop(columns=['marker']).columns.tolist()
    num_features = len(features)
    st.write(f"Total Number of Features: ", num_features)
    return

def display_features(data):
    st.write("List of All Features:")
    st.caption("*Expand to view list of Features.*")
    st.write(data.columns.tolist())

def target_variable(data):
    for column in data.select_dtypes(include=['object']):
        st.write("Target Feature: ", column)
        st.write('\nUnique Values for', column)
        st.dataframe(data[column].value_counts(), use_container_width=True)

def feature_importance(X_train, y_train): 
    f_scores, p_values = f_classif(X_train, y_train)
    X = data.drop("marker", axis=1)
    y = data["marker"]

    features_df = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores})
    features_df = features_df.sort_values('F-Score', ascending=False)

    return features_df

def explore_data(data):
    st.subheader("Dataset")

    with st.expander("Display Dataset Head"):
        st.write("")
        st.write(data.head())
    
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Basic Dataset Information",expanded=True):
            st.write("")
            st.write(f"Total Number of Data: ", data.shape[0])

            num_variables = len(data.columns)
            st.write(f"Total Number of Variables: ", num_variables)

            get_feature_count(data)

        with st.expander("Feature Data Types", expanded=True):
            data_types = data.dtypes.value_counts().rename_axis('data_type').reset_index(name='count')
            data_types.columns = ["data_type", "count"]
            data_types['data_type'] = data_types['data_type'].replace({'int64': 'Integer', 'float64': 'Float', 'object': 'Categorical'})

            st.write("")
            st.write("")
            st.dataframe(data_types, use_container_width=True)
            st.write("")
            st.write("")
            chart = alt.Chart(data_types.reset_index()).mark_bar().encode(x=alt.X('data_type', title='Data Type'), y=alt.Y('count', title='Number of Variables'))
            st.altair_chart(chart, use_container_width=True)

        with st.expander("Features",expanded=True):
            st.write("")
            display_features(data)
            st.markdown("***")

            st.write("Features Explanation: ")
            feature_table = Image.open('feature_table.png')
            st.image(feature_table, caption="Features Explanation")

            st.markdown("***")
            target_variable(data)
            

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
            st.write("Features Ranked based on ANOVA F-value.")
            
            features_df = feature_importance(X_train, y_train)

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

data, X_train, y_train = load_data()
explore_data(data)
