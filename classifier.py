import pandas as pd
import numpy as np
import datetime as dt
import os
import os.path
from pathlib import Path
import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

dir_ = Path('firedataset')
png_filepaths = list(dir_.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], png_filepaths))

filepaths = pd.Series(png_filepaths, name = 'File').astype(str)
labels = pd.Series(labels, name = 'Label')

df = pd.concat([filepaths, labels], axis=1)

df['Label'].replace({"non_fire_images":"nofire","fire_images":"fire"}, inplace=True)

vc = df['Label'].value_counts()
df = df.sample(frac = 1, random_state = 83).reset_index(drop = True)

train_df, test_df = train_test_split(df, train_size = 0.9, random_state = 86)

print('Training Dataset:')

print(f'Number of images: {train_df.shape[0]}')

print(f'Number of fire images: {train_df["Label"].value_counts()[0]}')
print(f'Number of non-fire images: {train_df["Label"].value_counts()[1]}\n')
      
print('Testing dataset:')
      
print(f'Number of testing images: {test_df.shape[0]}')
print(f'Images with fire: {test_df["Label"].value_counts()[0]}')
print(f'Without fire: {test_df["Label"].value_counts()[1]}')

LE = LabelEncoder()

y_test = LE.fit_transform(test_df["Label"])
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.1,
                                    rotation_range = 20,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    validation_split = 0.1)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    x_col = "File",
    y_col = "Label",
    target_size = (250, 250),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 32,
    shuffle = True,
    seed = 1,
    subset = "training")

validation_set = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    x_col = "File",
    y_col = "Label",
    target_size = (250, 250),
    color_mode ="rgb",
    class_mode = "binary",
    batch_size = 32,
    shuffle = True,
    seed = 1,
    subset = "validation")

test_set = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    x_col = "File",
    y_col = "Label",
    target_size = (250, 250),
    color_mode ="rgb",
    class_mode = "binary",
    shuffle = False,
    batch_size = 32)

CNN = Sequential()
CNN.add(Conv2D(32, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Conv2D(32, (3, 3), activation = 'relu'))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Conv2D(64, (3, 3), activation = 'relu'))
CNN.add(SpatialDropout2D(0.2))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Conv2D(128, (3, 3), activation = 'relu'))
CNN.add(SpatialDropout2D(0.4))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Flatten())
CNN.add(Dense(units = 256, activation = 'relu'))
CNN.add(Dropout(0.4))
CNN.add(Dense(units = 1, activation = 'sigmoid'))
# Callbacks
callbacks = [EarlyStopping(monitor = 'loss', mode = 'min', patience = 20, restore_best_weights = True)]
# Compile
CNN.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train
CNN_model = CNN.fit(training_set, epochs = 10, validation_data = validation_set, callbacks = callbacks)

acc = CNN_model.history['accuracy']
val_acc = CNN_model.history['val_accuracy']
loss = CNN_model.history['loss']
val_loss = CNN_model.history['val_loss']
epochs = range(1, len(acc) + 1)

CNN.save(os.path.join(os.path.dirname(__file__),"models","train_model.h5"))


    
    
    
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()