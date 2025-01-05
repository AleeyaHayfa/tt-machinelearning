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
  x_raw = df.drop('species', axis=1)
  x_raw

  st.write('Y')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data prepration (user input)
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgesen'))
  gender = st.selectbox('Gender', ('male', 'female'))
  # 32.1 mininum, 59.6 maximum, 43.9 average (same as other)
  bill_length_mm = st.slider('Bill length(mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth(mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0) 
  #"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"

 # user input into dataframe
# Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, x_raw], axis=0) #combine input features with penguin features

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**') #combine original dgn user input
  input_penguins

# Encode X 
encode = ['island', 'sex'] #combine island and island name pastu classify 0 or 1 for each particular name
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)
y
y_raw #compare dgn original nk tgk betul ke tak

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y
