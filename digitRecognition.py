from tkinter import *
import tkinter as tk
from keras.models import load_model
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk,Image
import PIL.Image
import io
import os

os.environ["PATH"] += ':/Users/jerrylai/Desktop/digitRecognition/virt/lib/python3.8/site-packages'

model = load_model("deepLearningModel.model")

window = tk.Tk()
window.title("Digit Recognition")
window.geometry("460x460")

def writeDigit(event):
    x = event.x
    y = event.y
    r=8
    canvas.create_oval(x, y, x+r, y+r, fill="black")
    predictButton.configure(state=NORMAL)

def makePrediction():
    img = cv2.imread('2.png')
    img = np.array(img)
    img = cv2.resize(img, (28, 28))
    
    pred = model.predict(img)
    print(np.argmax(pred))
    """ps = canvas.postscript(colormode="color")
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save("rgeg.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)

    img = np.reshape(1, 28, 28, 1)
    img = img/255.0

    result = model.predict(img)
    print(np.argmax(result))"""


def clearDigit():
    predictButton.configure(state=DISABLED)
    canvas.delete('all')

predictButton = tk.Button(window, text="Predict", width=15, bg="orange", fg='black', font=('times', 18, 'bold'), command=makePrediction, state=DISABLED)
predictButton.place(x=16, y=100)

clearButton = tk.Button(window, text="Clear", width=15, bg="orange", fg="black", font=('times', 18, 'bold'), command=clearDigit)
clearButton.place(x=16, y= 160)

canvas = tk.Canvas(window, width=260, height=260, cursor="pencil", highlightthickness=1, highlightbackground="red")
canvas.place(x=160, y=60)
canvas.bind("<B1-Motion>", writeDigit)

title = tk.Label(window, text="Hand Written Digits Detector", fg="black")
title.place(x=136,y=20)

window.mainloop()

