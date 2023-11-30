## Genel Kütüphaneler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

## Derin Öğrenme Kütüphaneleri

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

# Resmin boyutu: 48*48 piksel
picture_size = 48

folder_path = "C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/train"

expression = 'sad'

plt.figure(figsize= (12,12))
for i in range(1, 10, 1):
    plt.subplot(3,3,i)
    img = load_img(folder_path+ '/' +expression+"/"+
                  os.listdir(folder_path+ '/' + expression)[i], target_size=(picture_size, picture_size))
    plt.imshow(img)
plt.show()
#Çıktı https://puu.sh/JV9vP/edaf3d28bf.png

## Farklı duygu sınıflarını tanımlıyoruz
num_classes = 7

## Görüntü boyutunu tanımlıyoruz
img_rows,img_cols = 48,48

## Batch'i tanımlıyoruz
batch_size = 64

train_datagen = ImageDataGenerator(
                    rescale=1./255, #eğitim verilerindeki görüntülerin piksel değerlerini 0 ile 1 arasında yeniden boyutlandırır.
                    rotation_range=30, #eğitim verilerindeki görüntüleri rastgele 30 derece döndürür.
                    shear_range=0.3, #eğitim verilerindeki görüntüleri rastgele genişliklerinin veya yüksekliklerinin 0,3 katı kadar eğir
                    zoom_range=0.3, #eğitim verilerindeki görüntüleri rastgele orijinal boyutlarının 0,3 katına kadar yakınlaştırır veya uzaklaştırır
                    width_shift_range=0.4,
                    height_shift_range=0.4,
                    horizontal_flip=True,
                    fill_mode='nearest')
#Bu satır eğitim verileri için bir ImageDataGenerator nesnesi oluşturur. ImageDataGenerator sınıfı, görüntüleri yeniden boyutlandırma, döndürme, eğme, yakınlaştırma, kaydırma, çevirme ve eksik pikselleri doldurma gibi çeşitli yöntemler sunar.
##########

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',
                    target_size=(img_rows,img_cols),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            color_mode='grayscale',
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=True)


#train_datagen ve validation_datagen eğitim ve doğrulama veri kümelerini hazırlamak için kullanılan ImageDataGenerator nesneleridir. 
#train_generator ve validation_generator eğitim ve doğrulama veri kümelerini içeren veri oluşturucularıdır. 

model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block-1'den Block-5'e: Bunlar evrişim (convolutional) bloklarıdır, her biri evrişim katmanları, aktivasyon fonksiyonları (ELU), parti normalleştirme, en büyük havuzlama (max-pooling) ve bırakma (dropout) katmanları içerir. Evrişim katmanları giriş görüntülerinde özellikleri tespit eder.
######################

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

#Block-6 ve Block-7: Bunlar tam bağlantılı (dense) katmanlardır ve aktivasyon fonksiyonları, parti normalleştirme ve bırakma katmanları içerir. Bu katmanlar, evrişim katmanları tarafından öğrenilen uzaysal bilgileri birleştirir.

#Çıkış katmanı çok sınıflı sınıflandırma problemleri için uygun olan softmax aktivasyon fonksiyonuna sahiptir.

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr] #Geribildirimler (Callbacks):
#ModelCheckpoint: Doğrulama kaybına dayanarak en iyi modeli kaydeder.
#EarlyStopping: 3 epoch boyunca doğrulama kaybında iyileşme olmazsa eğitimi durdurur ve en iyi ağırlıkları geri yükler.
#ReduceLROnPlateau: Doğrulama kaybı durgunsa öğrenme hızını azaltır.



model.compile(loss='categorical_crossentropy', #Model, kategorik çapraz entropi kaybını, 0.001 öğrenme hızına sahip Adam optimizer'ı ve doğruluk metriğini kullanarak derlenir.
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])


nb_train_samples = 24320
nb_validation_samples = 3072
#nb_train_samples ve nb_validation_samples, eğitim ve doğrulama örnek sayısını temsil eder.
epochs=40
#epochs 40 olarak ayarlanmıştır.
