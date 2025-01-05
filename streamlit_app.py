import streamlit as st
import pandas as pd

st.title('ThinkTankers ML App')

st.write('This app builds a machine learning model')

# nanti boleh tukar dataset kat sini
with st.expander('Data'): #expander tu drop down button
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('Y')
  Y = df.species
  Y
 
