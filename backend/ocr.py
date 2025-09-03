import pytesseract
import easyocr
import cv2
from typing import List, Tuple

class OCREngine:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'])
    
    def extract_text_tesseract(self, image) -> str:
        """Extract text using Tesseract OCR"""
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    
    def extract_text_easyocr(self, image) -> str:
        """Extract text using EasyOCR"""
        results = self.easyocr_reader.readtext(image)
        if results:
            return ' '.join([result[1] for result in results])
        return ""
    
    def extract_text_combined(self, image) -> str:
        """Combine results from multiple OCR engines"""
        tesseract_result = self.extract_text_tesseract(image)
        easyocr_result = self.extract_text_easyocr(image)
        
        # Return the longer result (usually more accurate)
        if len(tesseract_result) >= len(easyocr_result):
            return tesseract_result
        return easyocr_result

