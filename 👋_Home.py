import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)

st.title("Network Intrusion Detection Model based on Power System Attack Datasets using Logistic Regression Algorithm")

st.markdown("***")
st.markdown(""" 

The Network Intrusion Detection Model is developed to provide classification of power system attack (Data Injection Attack) occurrence 
through Machine Learning implementation of Logistic Regression Algorithm.

#### Dataset Information
- The Dataset is an open source simulated power system dataset provided by Mississippi State University Oak Ridge National 
Laboratories which contains three datasets that comprise measurements of the regular, disturbed, controlled, and cyberattack 
behaviours of electric transmission systems whereby the collection contains measurements from relays, a simulated control panel, 
synchrophasor measurements, and data logs from Snort.

- Source: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
- More Information on Dataset: http://www.ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset_README.pdf 
- The Dataset is filtered to only include Natural Events and Data Injection Attack Events.
- Filtered Dataset: https://drive.google.com/drive/folders/1Y6PaJLt9P-GnD8hH9EQgDzLFdV8i0lmY?usp=share_link """)


st.write("")
st.markdown(""":point_left: **Select a module from the sidebar** to see and utilize functions available in the dashboard for the 
intrusion detection model.

#### Dashboard Module 
- Data Visualization
- Machine Learning Model Information
- Power System Attack Prediction  """)

st.sidebar.success("Select a module above.")