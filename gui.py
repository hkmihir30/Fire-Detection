import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import cv2
import os
from os.path import dirname
from tensorflow.keras.models import load_model

global model
model_file_path = os.path.join(dirname(__file__), 'models','train_model.h5')
model  = load_model(model_file_path)
print("Loaded model from disk")





classes = {0: 'Fire', 
           1: 'No Fire'}

def load_image(data_path):
    img = cv2.imread(data_path)
    img = cv2.resize(img, (250, 250))
    img = img/255
    
    return img

def predict(image_path):
    img = load_image(image_path)
    probabilities = model.predict(np.asarray([img]))[0]
    print(probabilities)
    if probabilities[0]<=0.50:
        class_idx = 0
        return {classes[class_idx]: 1-probabilities[0]}
    else:
        class_idx = 1    
        return {classes[class_idx]: probabilities[0]}

top=tk.Tk()
top.geometry('800x600')
top.title('Fire Detection')
top.configure(background='#CDCDCD')
label=tk.Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = tk.Label(top)


def classify(file_path):
    global label_packed
    prediction = predict(file_path)
    sign = "PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0])
    print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=tk.Button(top,text="Classify Image", command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=tk.Button(top,text="Upload an image",command=upload_image, padx=10,pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side=tk.BOTTOM,pady=50)
sign_image.pack(side=tk.BOTTOM,expand=True)
label.pack(side=tk.BOTTOM,expand=True)
heading = tk.Label(top, text="Fire Detector",pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()