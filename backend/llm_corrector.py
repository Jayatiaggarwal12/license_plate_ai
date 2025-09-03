import openai
import re
from typing import Optional

class LLMCorrector:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def correct_license_plate(self, ocr_text: str, image_description: str = "") -> str:
        """Use LLM to correct and validate license plate text"""
        
        prompt = f"""
        You are an expert at correcting license plate text from OCR. 
        
        OCR Result: "{ocr_text}"
        Image Context: {image_description}
        
        Rules for license plates:
        - Usually 6-10 characters
        - Mix of letters and numbers
        - Common OCR errors: 0/O, 1/I/l, 5/S, 8/B, 6/G
        - No special characters except hyphens
        
        Provide the most likely correct license plate text. If unsure, provide your best guess.
        Return only the corrected text, nothing else.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def validate_and_format(self, text: str) -> str:
        """Clean and format the license plate text"""
        # Remove unwanted characters
        cleaned = re.sub(r'[^A-Za-z0-9\-]', '', text)
        # Convert to uppercase
        return cleaned.upper()