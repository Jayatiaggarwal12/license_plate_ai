import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

class LicensePlateDetector:
    def __init__(self, model_path='models/license_plate_yolov8.pt'):
        self.model = YOLO(model_path)
    
    def detect_license_plate(self, image_path):
        """Detect license plate in image and return cropped plate"""
        results = self.model(image_path)
        
        if len(results[0].boxes) > 0:
            # Get the best detection (highest confidence)
            box = results[0].boxes[0]
            confidence = box.conf.item()
            
            if confidence > 0.5:  # Confidence threshold
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Load original image and crop license plate
                image = cv2.imread(image_path)
                cropped_plate = image[int(y1):int(y2), int(x1):int(x2)]
                
                return cropped_plate, confidence
        
        return None, 0.0
