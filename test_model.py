from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ModelTester:
    def __init__(self, model_path='models/license_plate_yolov8.pt'):
        """Initialize the model tester"""
        if not os.path.exists(model_path):
            # Try alternative path from training
            model_path = 'runs/detect/license_plate_detector/weights/best.pt'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = YOLO(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
    def test_single_image(self, image_path, save_result=True):
        """Test model on a single image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        print(f"🔍 Testing on: {image_path}")
        
        # Run inference
        results = self.model(image_path)
        
        # Get results
        result = results[0]
        
        if len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                print(f"  Detection {i+1}:")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                
            # Save annotated image
            if save_result:
                annotated = result.plot()
                output_path = f"test_results/result_{os.path.basename(image_path)}"
                os.makedirs("test_results", exist_ok=True)
                cv2.imwrite(output_path, annotated)
                print(f"  ✅ Saved result to: {output_path}")
                
            return result
        else:
            print("  ❌ No license plates detected")
            return None
    
    def test_validation_set(self):
        """Test on validation images"""
        val_images_dir = "data/valid/images"
        
        if not os.path.exists(val_images_dir):
            print(f"❌ Validation directory not found: {val_images_dir}")
            return
            
        image_files = [f for f in os.listdir(val_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("❌ No validation images found")
            return
            
        print(f"🧪 Testing on {len(image_files)} validation images...")
        
        detections = 0
        total_confidence = 0
        
        for img_file in image_files[:10]:  # Test first 10 images
            img_path = os.path.join(val_images_dir, img_file)
            result = self.test_single_image(img_path, save_result=True)
            
            if result and len(result.boxes) > 0:
                detections += 1
                total_confidence += result.boxes[0].conf.item()
        
        detection_rate = (detections / len(image_files[:10])) * 100
        avg_confidence = total_confidence / max(detections, 1)
        
        print(f"\n📊 Validation Results:")
        print(f"   Detection rate: {detection_rate:.1f}%")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Results saved in: test_results/")
        
    def benchmark_model(self):
        """Run model benchmarks"""
        print("⏱️ Running model benchmarks...")
        
        # Test inference speed
        test_image = "data/valid/images/" + os.listdir("data/valid/images")[0]
        
        import time
        times = []
        
        for _ in range(10):
            start = time.time()
            self.model(test_image)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        fps = 1 / avg_time
        
        print(f"📈 Performance Metrics:")
        print(f"   Average inference time: {avg_time:.3f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Model size: {os.path.getsize(self.model.ckpt_path) / (1024*1024):.1f} MB")

def main():
    print("🧪 YOLOv8 License Plate Model Testing")
    print("=" * 50)
    
    try:
        # Initialize tester
        tester = ModelTester()
        
        # Test on validation set
        tester.test_validation_set()
        
        # Run benchmarks
        tester.benchmark_model()
        
        print("\n✅ Testing completed!")
        print("Check 'test_results/' folder for annotated images")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        
        # Try to find the model in different locations
        possible_paths = [
            'models/license_plate_yolov8.pt',
            'runs/detect/license_plate_detector/weights/best.pt',
            'runs/detect/train/weights/best.pt'
        ]
        
        print("\n🔍 Looking for trained model in:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ Found: {path}")
            else:
                print(f"   ❌ Not found: {path}")

if __name__ == "__main__":
    main()