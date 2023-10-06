import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochs.h5')
image=cv2.imread('C:\\Users\\HP\\Downloads\\archive\\pred\\pred8.jpg')

img = Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)
inp_img=np.expand_dims(img,axis=0)

result=(model.predict(inp_img) > 0.5).astype("int32")
#result=model.predict_classes(inp_img)
print(result[0][0])