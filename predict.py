# -*- coding: utf-8 -*-
import sys
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

    
def path_img_predict(model,path):
    # Load Model
    model = load_model(model)
    img = load_img(path,color_mode = "grayscale",target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 1, 28, 28)/255
    predict = model.predict_classes(img)
    prob = model.predict_proba(img)
    print('Prediction:', predict)
    print('Prob:',prob)
    imgplot = plt.imshow(load_img(path))



# path_img_predict('./CNN_Mnist.h5','test5.jpg')

if __name__ == "__main__":
    path_img_predict(sys.argv[1],sys.argv[2])
