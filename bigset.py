# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 19:06:49 2021

@author: USER
"""


import streamlit as st
from keras.models import load_model
import cv2
from PIL import *
import numpy as np


st.title("Brain MRI classifier")

uploaded_file = st.file_uploader("Upload Files",type=['png','jpg'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)

classifier=load_model('C:/Users/USER/Documents/python projects/brain tumour/big dataset/brainMRImodel.h5')

tumor_labels=['glioma tumor','no tumor','meningioma','pituitary tumor']





def my_widget(key):
    st.subheader('Hello there!')
    st.write('upload your brain MRI and the app will return the classified label whether there is tumor or what type of tumor it is.')
with st.sidebar:
    clicked = my_widget("third")


image = Image.open(uploaded_file)
col1, col2 = st.beta_columns(2)
col1.image(image, caption='Input',use_column_width=True)



def analyse(a):
    image=a
    
    a_image=np.array(image)
    p_image=cv2.cvtColor(a_image, cv2.COLOR_RGB2BGR)
    r_image=cv2.resize(p_image, (124,124))
    img = r_image.reshape(1,124,124,3)

    prediction=classifier.predict(img)
    label=tumor_labels[prediction.argmax()]
    return(label)
    
    

if col2.button('find result'):
    col2.header(analyse(image))