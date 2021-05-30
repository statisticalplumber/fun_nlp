# loading packages
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

st.sidebar.markdown('<h3> App Test </h3>', unsafe_allow_html = True)

st.title("App is use")

# reading csv and display

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.DataFrame(load_iris().data, columns=['sl', 'sw', 'pl', 'pw'])
    return df

df = load_data()
# display data
st.write(df.head())
