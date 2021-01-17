from tkinter import *
import tkinter as tk
from keras.models import load_model
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk,Image
import PIL.Image
import io

model = load_model("digit.h5")

window = tk.Tk()
window.title("Digit Recognition")
window.geometry("460x460")

def writeDigit(event):
    x = event.x
    y = event.y
    r=16
    canvas.create_oval(x, y, x+r, y+r, fill="black")
    predictButton.configure(state=NORMAL)

def makePrediction():
    
    global accLabel, resLabel

    ps = canvas.postscript(colormode="color")
    img = Image.open(io.BytesIO(ps.encode('utf-8')))

    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)

    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0

    result = model.predict(img)

    resLabel = tk.Label(window, text=f"Predicted result: {np.argmax(result)}", fg="black", width=32, height=1, bg="lime", font=("times", 16, "bold"))
    resLabel.place(x=160, y=328)

    accLabel = tk.Label(window, text=f"Accuracy: {result[0][np.argmax(result)]}", fg="black", width=32, height=1, bg="lime", font=("times", 16, "bold"))
    accLabel.place(x=160, y=348)



def clearDigit():
    predictButton.configure(state=DISABLED)
    canvas.delete('all')
    accLabel.destroy()
    resLabel.destroy()

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

