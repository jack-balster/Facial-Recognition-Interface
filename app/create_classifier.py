import numpy as np
from PIL import Image
import os, cv2
import sqlite3

# Method to train custom classifier to recognize face
def train_classifer(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd(), "data", name)

    faces = []
    ids = []
    pictures = []

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    for root, dirs, files in os.walk(path):
        pictures = files

    for pic in pictures:
        imgpath = os.path.join(path, pic)
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        
        # Extract the numeric part before the underscore
        try:
            id = int(pic.split('_')[0])
        except ValueError:
            print(f"Skipping file {pic}: unable to extract ID")
            continue
        
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write(f"./data/classifiers/{name}_classifier.xml")
    
    # Update the database with the number of images
    update_num_of_images(name, len(faces))

def update_num_of_images(name, num_of_images):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("UPDATE users SET num_of_images = ? WHERE name = ?", (num_of_images, name))
    conn.commit()
    conn.close()
