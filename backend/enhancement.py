import cv2
import numpy as np
from PIL import Image, ImageEnhance

class ImageEnhancer:
    @staticmethod
    def enhance_license_plate(image):
        """Apply multiple enhancement techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Resize for better OCR
        height, width = sharpened.shape
        new_height = max(100, height * 3)  # Minimum height of 100px
        new_width = int(width * (new_height / height))
        resized = cv2.resize(sharpened, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return resized