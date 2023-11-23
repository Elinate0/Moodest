## General Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

## Deep Learning Libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array


df = pd.read_csv("C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/fer2013.csv")

df.head()

df.shape

df.isnull().sum()

df['emotion'].value_counts()

train_data_dir = "C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/train"
validation_data_dir = "C:/Users/basak/OneDrive/Masaüstü\OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/test"

# size of the image: 48*48 pixels
picture_size = 48

# input path for the images
folder_path = "C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/train"

expression = 'sad'

plt.figure(figsize= (12,12))
for i in range(1, 10, 1):
    plt.subplot(3,3,i)
    img = load_img(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i], target_size=(picture_size, picture_size))
    plt.imshow(img)
plt.show()

## Defining different classes of emotion
num_classes = 7

## Define image size
img_rows,img_cols = 48,48

## Define the batch
batch_size = 64