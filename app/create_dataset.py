import cv2
import os
import sqlite3

# Constants
IMAGE_COUNT_LIMIT = 300  # Maximum number of images to capture
CASCADE_FILE = "./data/haarcascade_frontalface_default.xml"  # Path to the Haar Cascade file for face detection
OUTPUT_DIR = "./data/"  # Directory where captured images will be stored

def update_num_of_images(name, num_of_images):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("UPDATE users SET num_of_images = ? WHERE name = ?", (num_of_images, name))
    conn.commit()
    conn.close()
    
def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):  # Check if the directory already exists
        os.makedirs(path)  # Create the directory if it does not exist

def capture_faces(detector, img, path, num_of_images, name):
    """
    Detect and save faces from the image.

    Parameters:
        detector (cv2.CascadeClassifier): The face detector.
        img (numpy.ndarray): The image in which to detect faces.
        path (str): The directory path to save captured images.
        num_of_images (int): The current count of captured images.
        name (str): The name of the person being captured.

    Returns:
        tuple: The processed image and updated number of captured images.
    """
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    faces = detector.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=5)  # Detect faces
    
    for x, y, w, h in faces:  # Iterate over each detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a rectangle around the face
        cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))  # Label the rectangle
        cv2.putText(img, f"{num_of_images} images captured", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))  # Display image count
        new_img = img[y:y+h, x:x+w]  # Crop the face from the image
        cv2.imwrite(os.path.join(path, f"{num_of_images}_{name}.jpg"), new_img)  # Save the cropped face as .jpg file
        num_of_images += 1  # Increment the count of captured images
    
    return img, num_of_images  # Return the processed image and the updated count of images

def start_capture(name):
    """
    Capture images from the webcam and save detected faces.

    Parameters:
        name (str): The name of the person being captured.

    Returns:
        int: The number of images captured.
    """
    path = os.path.join(OUTPUT_DIR, name)
    create_directory(path)
    
    detector = cv2.CascadeClassifier(CASCADE_FILE)
    vid = cv2.VideoCapture(0)
    num_of_images = 0
    
    while True:
        ret, img = vid.read()
        if not ret:
            print("Failed to capture image")
            continue
        
        img, num_of_images = capture_faces(detector, img, path, num_of_images, name)
        cv2.imshow("Face Detection", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27 or num_of_images >= IMAGE_COUNT_LIMIT:
            break
    
    vid.release()
    cv2.destroyAllWindows()
    update_num_of_images(name, num_of_images)
    return num_of_images

def take_video(name, video):
    """
    Capture images from a video file and save detected faces.

    Parameters:
        name (str): The name of the person being captured.
        video (str): The path to the video file.

    Returns:
        int: The number of images captured.
    """
    path = os.path.join(OUTPUT_DIR, name)  # Define the directory path for the captured images
    create_directory(path)  # Create the directory if it doesn't exist
    
    detector = cv2.CascadeClassifier(CASCADE_FILE)  # Load the face detector
    vid = cv2.VideoCapture(video)  # Open the video file
    if not vid.isOpened():  # Check if the video file was opened successfully
        print("Error: Could not open video file.")
        return 0  # Return 0 if the video file could not be opened
    
    num_of_images = 0  # Initialize the count of captured images
    
    while True:
        ret, img = vid.read()  # Capture a frame from the video
        if not ret:  # Check if the frame was captured successfully
            break  # Exit the loop if no more frames are available
        
        img, num_of_images = capture_faces(detector, img, path, num_of_images, name)  # Detect and save faces
        
        cv2.imshow("Face Detection", img)  # Display the image with annotations
        
        key = cv2.waitKey(1) & 0xFF  # Check for key press
        if key == ord("q") or key == 27 or num_of_images >= IMAGE_COUNT_LIMIT:  # Exit conditions
            break
    
    vid.release()  # Release the video file
    cv2.destroyAllWindows()  # Close all OpenCV windows
    return num_of_images  # Return the number of images captured
