import cv2
from time import time
from PIL import Image
from tkinter import messagebox
import os
import sqlite3

# Detector class for recognizing face of existing (selected) user
def face_recognition(name, timeout=5):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    
    # Create an LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Construct the path to the classifier file for the specified user
    classifier_path = f"./data/classifiers/{name}_classifier.xml"
    
    # Check if the classifier file exists
    if not os.path.exists(classifier_path):
        # Show an error message if the file does not exist
        messagebox.showerror("Error", f"Classifier file for {name} not found. Please train the model first")
        return

    try:
        # Try to read the classifier file
        recognizer.read(classifier_path)
    except cv2.error as e:
        # Show an error message if reading the file fails
        messagebox.showerror("Error", f"Failed to read classifier file: {e}")
        return

    # Open the default camera
    cap = cv2.VideoCapture(0)
    pred = False
    start_time = time()
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            # Show an error message if reading the frame fails
            messagebox.showerror("Error", "Failed to capture video frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Get the region of interest (ROI) for the detected face
            roi_gray = gray[y:y+h, x:x+w]
            
            # Predict the ID and confidence of the detected face
            id, confidence = recognizer.predict(roi_gray)
            confidence = 100 - int(confidence)
            
            if confidence > 50:
                # If the confidence is greater than 50%, mark the face as recognized
                pred = True
                text = 'Recognized: ' + name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2  # Increase the font scale for larger text
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y - 10), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # If the confidence is 50% or lower, mark the face as unknown
                pred = False
                text = "Unknown Face"
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2  # Increase the font scale for larger text
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y - 10), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame with the detected face and recognition result
        cv2.imshow("image", frame)

        # Calculate the elapsed time
        elapsed_time = time() - start_time
        
        if elapsed_time >= timeout:
            # If the elapsed time exceeds the timeout, show the result
            print(pred)
            if pred:
                messagebox.showinfo('Congrat', 'You have been recognized and are logged in')
            else:
                messagebox.showerror('Alert', 'Unrecognized, Please try again with better lighting or switch users')
            break

        # Check if the 'q' key was pressed to quit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
