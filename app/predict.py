import cv2
import os

def predict(name, sample):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    
    # Create an LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Construct the path to the classifier file for the specified user
    classifier_path = f"./data/classifiers/{name}_classifier.xml"
    
    if not os.path.exists(classifier_path):
        print(f"Classifier file for {name} not found. Please train the model first.")
        return
    
    try:
        recognizer.read(classifier_path)
    except cv2.error as e:
        print(f"Failed to read classifier file: {e}")
        return
    
    # Open the video sample
    cap = cv2.VideoCapture(sample)
    pred = False
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
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
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y - 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # If the confidence is 50% or lower, mark the face as unknown
                pred = False
                text = "Unknown Face"
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y - 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        print(pred)
        cv2.imshow("image", frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
# predict('tho', r'data/WIN_20230920_07_56_11_Pro.mp4')
