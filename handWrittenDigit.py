from cv2 import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("digit.model")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey, 126, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    for num in contours:
        x, y, w, h = cv2.boundingRect(num)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        digit = thresh[y:y+h, x:x+w]

        resized = cv2.resize(digit, (18,18))

        padded = np.pad(resized, ((5, 5),(5, 5)), 'constant', constant_values=0)

        #res = model.predict(padded.reshape(1, 28, 28, 1))

        #print("{}".format(np.argmax(res)))


    cv2.imshow("Digit Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()