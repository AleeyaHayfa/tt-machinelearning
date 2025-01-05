import streamlit as st
import pandas as pd

st.title('ThinkTankers ML App')

st.write('This app builds a machine learning model')

# nanti boleh tukar clean letak kat sini
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

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data prepration
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgesen'))
  gender = st.selectbox('Gender', ('male', 'female'))
  # 32.1 mininum, 59.6 maximum, 43.9 average
  bill_length_mm = st.slider('Bill length(mm)', 32.1, 59.6, 43.9)
  #"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"

  
