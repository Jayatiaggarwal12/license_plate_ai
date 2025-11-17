# 🚗 Automatic License Plate Recognition System

A high-accuracy ALPR system using **YOLOv8** for detection, **EasyOCR** for recognition, and **Google Gemini** for intelligent correction. Achieves **93.6% end-to-end accuracy** with a modern web interface.

## ✨ Features

- **YOLOv8 Detection**: 96.3% accuracy in locating license plates
- **Multi-View OCR**: 7 preprocessing variants for robust recognition
- **AI-Powered Correction**: Gemini synthesizes multiple OCR candidates (97.2% accuracy)
- **Web Interface**: FastAPI backend + Streamlit frontend
- **Real-time Processing**: ~1.8 seconds per image
- **Batch Processing**: Handle multiple images efficiently

## 🏗️ Architecture

```
Input Image → YOLOv8 Detection → Multi-View Preprocessing → 
EasyOCR (7 variants) → Gemini AI Synthesis → Final Plate Number
```

## 🛠️ Tech Stack

- **Detection**: YOLOv8 (Ultralytics)
- **OCR**: EasyOCR
- **AI Correction**: Google Gemini API
- **Image Processing**: OpenCV, NumPy, Pillow
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Development**: Python 3.8+, Google Colab

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/alpr-system.git
cd alpr-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
YOLO_MODEL_PATH=path/to/best.pt
```

### 4. Download Model
Place your trained YOLOv8 model (`best.pt`) in the project directory.

## 🚀 Quick Start

### Run Backend (Terminal 1)
```bash
python main.py
# API runs at http://localhost:8000
```

### Run Frontend (Terminal 2)
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Test API
```bash
curl -X POST "http://localhost:8000/detect-plate" \
  -F "file=@vehicle_image.jpg"
```

## 📊 Performance

| Metric | Score |
|--------|-------|
| Detection Accuracy | 96.3% |
| OCR Accuracy (AI) | 97.2% |
| End-to-End Accuracy | 93.6% |
| Processing Speed | 1.8s/image |

## 🎯 Usage

### Web Interface
1. Open Streamlit app at `http://localhost:8501`
2. Upload vehicle image
3. Click "Detect License Plate"
4. View results with annotated image

### API Endpoint
```python
import requests

with open('vehicle.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect-plate',
        files={'file': f}
    )
    
result = response.json()
print(f"Plate: {result['plate_number']}")
print(f"Confidence: {result['confidence']}")
```

## 📁 Project Structure

```
alpr-system/
├── main.py              # FastAPI backend
├── app.py               # Streamlit frontend
├── best.pt              # Trained YOLOv8 model
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
└── README.md           # Documentation
```

## 🔧 Configuration

### Multi-View Preprocessing
The system generates 7 enhanced views:
- Grayscale
- CLAHE (contrast enhancement)
- Otsu thresholding
- Adaptive thresholding
- Sharpened
- Denoised
- Normalized

### API Parameters
- **Detection Confidence**: 0.5 (adjustable in `main.py`)
- **OCR Languages**: English
- **Gemini Temperature**: 0.1 (low for consistency)

## 🌟 Key Innovation

**Multi-Candidate AI Synthesis**: Unlike traditional single-pass OCR, our system:
1. Generates multiple preprocessed views
2. Runs OCR on each variant
3. Uses Gemini to intelligently synthesize the best result
4. Applies contextual correction (O↔0, I↔1, B↔8)

Result: **+29% accuracy improvement** over single OCR (68% → 97.2%)

## 🎓 Training Details


- **Training**: 75 epochs on Google Colab GPU- **Framework**: Ultralytics YOLOv8n (nano model)
- **Augmentation**: Mosaic, rotation, scaling, color jittering

## 🚧 Limitations

- Small/distant plates (<35×12 pixels): Lower accuracy
- Heavy occlusions (>60% coverage): May fail
- Unusual vintage plates: Limited recognition
- API dependency: Requires Gemini API access

## 🔮 Future Enhancements

- [ ] Vehicle tracking for video streams
- [ ] Custom OCR fine-tuning for Indian plates
- [ ] Local language model deployment
- [ ] Database integration for validation
- [ ] Multi-country plate support
- [ ] Edge device deployment

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **DIC UIET, Panjab University, Chandigarh**
- **Dr. Naveen Aggarwal** (Guide)
- Ultralytics (YOLOv8)
- JaidedAI (EasyOCR)
- Google (Gemini API, Colab)



---

⭐ **Star this repo** if you find it helpful!
