# pytorch_models.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

class FaceDetector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_faces(self, image):
        # Convert the image to RGB and apply the transformations
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image)

        # Process the predictions
        # Assume predictions are bounding boxes
        boxes = predictions[0]['boxes'].cpu().numpy()
        return boxes

class FaceRecognizer:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def recognize_face(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)

        return torch.argmax(output, dim=1).item()
