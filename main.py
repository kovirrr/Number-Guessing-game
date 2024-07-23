import cv2
import numpy as np
from ultralytics import YOLO
import time

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # This will be a list of (x, y, w, h) tuples

def detect_hairline(image):
    hairline_model = YOLO('/Users/koviressler/Desktop/DailyTefillin/hairline_detection/hairlineAI.pt')
    hairlines = hairline_model(image)
    #print("hairlines:", hairlines)
    return hairlines

def detect_tefillin(image):
    tefillin_model = YOLO('/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/runs/detect/train3/weights/best.pt')
    tefillins = tefillin_model(image)
    return tefillins


# Usage
cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break
    else:
        frame = cv2.flip(frame, 1) #horizontal (like a phone is)

    #detected_faces = detect_face(frame)#faces
    detected_hairlines = detect_hairline(frame)#hairlines

    #for (x, y, w, h) in detected_faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) #put box around faces

    #if len(detected_faces) > 0: #only check for hairlines if a face is detected
    for r in detected_hairlines:
            if r.masks is not None:
                for mask in r.masks.xy:
                    # Convert to integer coordinates
                    mask = mask.astype(np.int32)

                    # Draw the polygon
                    cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2) #put polygon around hairlines


    cv2.imshow('DailyTefillin', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()