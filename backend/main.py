from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import base64

from detection import LicensePlateDetector
from enhancement import ImageEnhancer
from ocr import OCREngine
from llm_corrector import LLMCorrector

app = FastAPI(title="License Plate Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = LicensePlateDetector()
enhancer = ImageEnhancer()
ocr_engine = OCREngine()
llm_corrector = LLMCorrector(api_key="YOUR_OPENAI_API_KEY")

@app.post("/detect-license-plate")
async def detect_license_plate(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temporarily for detection
        temp_path = f"temp_{file.filename}"
        cv2.imwrite(temp_path, image)
        
        # Step 1: Detect license plate
        plate_crop, confidence = detector.detect_license_plate(temp_path)
        
        if plate_crop is None:
            return JSONResponse({
                "success": False,
                "message": "No license plate detected",
                "confidence": 0.0
            })
        
        # Step 2: Enhance image
        enhanced_plate = enhancer.enhance_license_plate(plate_crop)
        
        # Step 3: OCR extraction
        ocr_text = ocr_engine.extract_text_combined(enhanced_plate)
        
        # Step 4: LLM correction
        corrected_text = llm_corrector.correct_license_plate(ocr_text)
        final_text = llm_corrector.validate_and_format(corrected_text)
        
        # Convert enhanced image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', enhanced_plate)
        enhanced_b64 = base64.b64encode(buffer).decode()
        
        return JSONResponse({
            "success": True,
            "license_plate": final_text,
            "raw_ocr": ocr_text,
            "confidence": confidence,
            "enhanced_image": f"data:image/jpeg;base64,{enhanced_b64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "License Plate Recognition API"}
