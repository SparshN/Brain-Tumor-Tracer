import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os 
import streamlit as st


#'C:\\Users\\HP\\Downloads\\archive\\pred\\pred0.jpg' for line 13

def pred(file_adr) :    
    model=load_model('BrainTumor10Epochs.h5')
    image=cv2.imread(file_adr)
    img = Image.fromarray(image)
    img=img.resize((64,64))
    img=np.array(img)
    inp_img=np.expand_dims(img,axis=0)
    result=(model.predict(inp_img) > 0.5).astype("int32")
    print(result[0][0])
    return result[0][0]
#result=model.predict_classes(inp_img)

st.title('Prediction of Brain tumor')
uploaded_image = st.file_uploader('Choose an image')


if uploaded_image is not None:
 with open(os.path.join("temp", uploaded_image.name), "wb") as f:
    f.write(uploaded_image.getbuffer())

    # Get the file address (path) of the uploaded image
    file_adr= os.path.abspath(os.path.join("C:\\Users\\HP\\Downloads\\archive\\temp", uploaded_image.name))
    output=pred(file_adr)
 display_image = Image.open(uploaded_image)
 col1,col2 = st.columns(2)

 with col1:
    st.header('Your uploaded image')
    st.image(display_image)
 with col2:
    st.header("Your result: ")
    if output==0:
      st.header(' No Brain Tumor Detected')
    else:
         st.header('  Yes,There is Brain Tumor')
