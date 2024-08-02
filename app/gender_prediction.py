import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model # type: ignore
from keras_preprocessing.image import img_to_array
import numpy as np

# Initialize MTCNN and load models
detector = MTCNN()  # Initialize MTCNN detector for face detection
emotion_model_path = "./data/_mini_XCEPTION.106-0.65.hdf5"  # Path to the emotion detection model
age_proto = "./data/age_deploy.prototxt"
age_model = "./data/age_net.caffemodel"
gender_proto = "./data/gender_deploy.prototxt"
gender_model = "./data/gender_net.caffemodel"

# Constants for model mean values and labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-3)', '(4-7)', '(8-12)', '(13-19)', '(20-25)', '(26-32)', '(34-40)', '(41-49)', '(50-59)', '(60-100)']
gender_list = ['Male', 'Female']
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load pre-trained models
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')  # Haar cascade for face detection
emotion_classifier = load_model(emotion_model_path, compile=False)  # Load emotion detection model
age_net = cv2.dnn.readNet(age_model, age_proto)  # Load age detection model
gender_net = cv2.dnn.readNet(gender_model, gender_proto)  # Load gender detection model

def age_and_gender_prediction():
    """Perform age and gender prediction using the webcam."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Failed to capture image")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.3, minNeighbors=5)  # Detect faces
        for (x, y, w, h) in faces:
            roi = rgb_frame[y:y + h, x:x + w]  # Region of interest for the detected face
            blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  # Create blob
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()  # Predict gender
            gender = gender_list[gender_preds[0].argmax()]
            age_net.setInput(blob)
            age_preds = age_net.forward()  # Predict age
            age = age_list[age_preds[0].argmax()]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face
            cv2.putText(frame, f"{gender}, {age} years", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0))  # Display gender and age

        # Display instruction text
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
        cv2.imshow("Gender and Age Prediction", frame)  # Show the frame with predictions
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def emotion_prediction():
    """Perform emotion prediction using the webcam."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Failed to capture image")
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)  # Detect faces
        for (x, y, w, h) in faces:
            roi = gray_frame[y:y + h, x:x + w]  # Region of interest for the detected face
            roi = cv2.resize(roi, (48, 48))  # Resize ROI to 48x48
            roi = roi.astype("float") / 255.0  # Normalize ROI
            roi = img_to_array(roi)  # Convert ROI to array
            roi = np.expand_dims(roi, axis=0)  # Expand dimensions for prediction
            preds = emotion_classifier.predict(roi)[0]  # Predict emotion
            label = emotions[preds.argmax()]  # Get the label of the predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face
            cv2.putText(frame, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0))  # Display emotion label

        # Display instruction text
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
        cv2.imshow("Emotion Detection", frame)  # Show the frame with predictions
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
