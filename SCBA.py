import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
from PIL import Image
import os
from config.definitions import ROOT_DIR

#ML model for predicting elastic buckling load
#Gradient boosting with categorical features support (CatBoost)
CatBoost_Elastic_model = joblib.load(os.path.join(ROOT_DIR, 'Cellular_Beams_Elastic_CatBoost_2021_03_20.joblib'))
CatBoost_Elastic_scaler=pickle.load(open(os.path.join(ROOT_DIR, 'Cellular_Beams_Elastic_CatBoost_2021_03_20.pkl'),'rb'))

#ML model for predicting ultimate load
#Gradient boosting with categorical features support (CatBoost)
CatBoost_Inelastic_model = joblib.load(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_CatBoost_2021_03_14.joblib'))
CatBoost_Inelastic_scaler=pickle.load(open(os.path.join(ROOT_DIR,'Cellular_Beams_Inelastic_CatBoost_2021_03_14.pkl'),'rb'))

st.header('Elastic Buckling and Ultimate Loads of Steel Cellular Beams Predicted by CatBoost Models')

st.sidebar.header('User Input Parameters')

image = Image.open(os.path.join(ROOT_DIR, 'Cell_Beam_App.png'))
st.subheader('Dimensional Parameters')
st.image(image)

def user_input_features():
    span_length = st.sidebar.slider('L (mm)', min_value=4000, max_value=7000, step=250)
    beam_height = st.sidebar.slider('H (mm)', min_value=420, max_value=700, step=20)
    flange_width = st.sidebar.slider('Bf (mm)', min_value=162, max_value=270, step=27)
    flange_thickness = st.sidebar.slider('Tf (mm)', min_value=15, max_value=25, step=5)
    web_thickness = st.sidebar.slider('Tw (mm)', min_value=9, max_value=15, step=3) 
    height_to_diameter = st.sidebar.slider('H/Do', min_value=1.25, max_value=1.70, step=0.05)
    spacing_to_diameter = st.sidebar.slider('So/Do', min_value=1.10, max_value=1.49, step=0.03)
    yield_strength_sel = st.sidebar.radio('Fy (MPa)', ('235','355','440')) 
    if yield_strength_sel=='235': yield_strength=235
    elif yield_strength_sel=='355': yield_strength=355
    elif yield_strength_sel=='440': yield_strength=440
    
    data = {'L (mm)': span_length,
            'H (mm)': beam_height,
            'Bf (mm)': flange_width,
            'Tf (mm)': flange_thickness,
            'Tw (mm)': web_thickness,           
            'H/Do': height_to_diameter,           
            'So/Do': spacing_to_diameter,
            'Fy (MPa)': yield_strength}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

L=df['L (mm)'].values.item()
H=df['H (mm)'].values.item()
FW=df['Bf (mm)'].values.item()
TF=df['Tf (mm)'].values.item()
TW=df['Tw (mm)'].values.item()
H_Do=df['H/Do'].values.item()
So_Do=df['So/Do'].values.item()
Do=H/H_Do
So=Do*So_Do
WP=Do*(So_Do-1)
N_holes=math.floor((L-2*Do)/So)
Led=0.5*(L-N_holes*So-Do)
Fy=df['Fy (MPa)'].values.item()

user_input={'L': "{:.0f}".format(L),
            'H': "{:.0f}".format(H),
            'Bf': "{:.0f}".format(FW),
            'Tf': "{:.0f}".format(TF),
            'Tw': "{:.0f}".format(TW),
            'Do': "{:.1f}".format(Do),
            'So': "{:.1f}".format(So),
            'WP': "{:.1f}".format(WP),
            'Led': "{:.1f}".format(Led),
            'Fy': "{:.0f}".format(Fy),
            'n': "{:.0f}".format(N_holes),
            'H/Do': "{:.2f}".format(H_Do),
            'So/Do': "{:.2f}".format(So_Do)}
user_input_df=pd.DataFrame(user_input, index=[0])
st.subheader('User Input Parameters')
st.write(user_input_df)

X_IP_Elastic=np.array([[L,H,Do,WP,FW,Led,TF,TW]])
X_IP_Inelastic=np.array([[L,H,Do,WP,FW,Fy,Led,TF,TW]])

X_IP_Elastic_CatBoost=CatBoost_Elastic_scaler.transform(X_IP_Elastic)
X_IP_Inelastic_CatBoost=CatBoost_Inelastic_scaler.transform(X_IP_Inelastic)

w_cr_CatBoost=CatBoost_Elastic_model.predict(X_IP_Elastic_CatBoost).item()
w_max_CatBoost=CatBoost_Inelastic_model.predict(X_IP_Inelastic_CatBoost).item()

st.subheader('CatBoost Model Predictions (kN/m)')

predictions={'Elastic Buckling Load': "{:.1f}".format(w_cr_CatBoost),
              'Ultimate Load': "{:.1f}".format(w_max_CatBoost)}
predictions_df=pd.DataFrame(predictions, index=[0])

def color_col (col):
    color = '#B4F6B6'
    return ['background-color: %s' % color]
	
st.dataframe(predictions_df.style.apply(color_col))

st.write('An appropriate safety factor should be applied to the predicted ultimate load.')

st.subheader('Nomenclature')
st.write('Fy is beam yield strength (MPa); n is number of evenly spaced openings along beam span; and CatBoost is gradient boosting with categorical features support.')

st.subheader('Reference')
st.write('Degtyarev, V.V., Tsavdaridis, K.D. Buckling and ultimate load prediction models for steel perforated beams using machine learning algorithms. Preprint')

