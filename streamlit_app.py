import streamlit as st
import pandas as pd

st.title('ThinkTankers ML App')

st.write('This app builds a machine learning model')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df
