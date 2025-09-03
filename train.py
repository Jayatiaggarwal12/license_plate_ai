from ultralytics import YOLO
import os
import yaml
import time

def verify_dataset_structure():
    """Verify that the local dataset structure is correct"""
    required_structure = {
        'data/data.yaml': 'Dataset configuration file',
        'data/train/images': 'Training images',
        'data/train/labels': 'Training labels',
        'data/test/images': 'Test images', 
        'data/test/labels': 'Test labels',
        'data/valid/images': 'Validation images',
        'data/valid/labels': 'Validation labels'
    }
    
    print("Checking dataset structure...")
    missing_items = []
    total_images = 0
    
    for path, description in required_structure.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"✅ {description}: {path}")
            else:
                file_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.txt'))]) if os.path.isdir(path) else 0
                print(f"✅ {description}: {path} ({file_count} files)")
                if 'images' in path:
                    total_images += file_count
        else:
            print(f"❌ Missing: {path}")
            missing_items.append(path)
    
    print(f"\n📊 Total training images: {total_images}")
    
    if missing_items:
        print(f"\n❌ Missing {len(missing_items)} required items")
        return False, 0
    
    print("✅ Dataset structure verified!")
    return True, total_images

def check_data_yaml():
    """Check and display data.yaml configuration"""
    try:
        with open('data/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("\nDataset configuration:")
        print(f"  Classes: {config.get('nc', 'Unknown')}")
        print(f"  Names: {config.get('names', 'Unknown')}")
        print(f"  Train path: {config.get('train', 'Unknown')}")
        print(f"  Validation path: {config.get('val', 'Unknown')}")
        print(f"  Test path: {config.get('test', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False

def optimize_training_params(total_images):
    """Optimize training parameters based on dataset size"""
    # Calculate optimal parameters for fast training
    if total_images < 50:
        epochs = 20
        batch_size = 4
        imgsz = 416
        patience = 5
        print("🚀 Small dataset detected - using fast training settings")
    elif total_images < 200:
        epochs = 30  # Reduced from 100
        batch_size = 8
        imgsz = 640
        patience = 8
        print("🚀 Medium dataset detected - using balanced settings")
    else:
        epochs = 50  # Still reduced from 100
        batch_size = 16
        imgsz = 640
        patience = 10
        print("🚀 Large dataset detected - using optimized settings")
    
    return epochs, batch_size, imgsz, patience

def train_model_fast(epochs, batch_size, imgsz, patience):
    """Train YOLOv8 model with optimized settings for speed"""
    try:
        print(f"\n🏋️ Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {imgsz}")
        print(f"   Early stopping patience: {patience}")
        print(f"   Device: CPU (change to 'cuda' for GPU)")
        
        print("\nInitializing YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Nano is fastest
        
        print("🚀 Starting fast training...")
        start_time = time.time()
        
        results = model.train(
            data='data/data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='license_plate_detector_fast',
            patience=patience,
            save=True,
            plots=True,    # Generate plots - FIXED: Removed duplicate
            device='cpu',  # Change to 'cuda' if you have GPU
            workers=2,     # Reduced workers for stability
            cache=False,   # Disable cache for memory efficiency
            verbose=True,
            # Performance optimizations
            amp=False,     # Disable mixed precision for CPU
            save_period=10, # Save less frequently
            val=True,      # Enable validation
            exist_ok=True  # Overwrite existing
        )
        
        training_time = time.time() - start_time
        print(f"\n⏱️ Training completed in: {training_time/60:.1f} minutes")
        
        # Create models directory and save
        os.makedirs('models', exist_ok=True)
        
        # The best model is automatically saved
        best_model_path = 'runs/detect/license_plate_detector_fast/weights/best.pt'
        final_model_path = 'models/license_plate_yolov8_fast.pt'
        
        # Copy best model to models folder
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Best model copied to: {final_model_path}")
        
        # Print training results
        if hasattr(results, 'results_dict'):
            print(f"\n📊 Training Results:")
            print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   Final loss: {results.results_dict.get('train/box_loss', 'N/A')}")
        
        return final_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Quick fixes:")
        print("1. Reduce batch size: try batch=2")
        print("2. Reduce image size: try imgsz=320")
        print("3. Check RAM usage")
        print("4. Verify all image files are valid")
        return None

def quick_test_model(model_path):
    """Quick test of the trained model"""
    try:
        print(f"\n🧪 Quick model test...")
        model = YOLO(model_path)
        
        # Test on a validation image
        val_images_dir = "data/valid/images"
        if os.path.exists(val_images_dir):
            test_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if test_images:
                test_image = os.path.join(val_images_dir, test_images[0])
                print(f"Testing on: {test_image}")
                
                results = model(test_image)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    confidence = results[0].boxes[0].conf.item()
                    print(f"✅ Detection successful! Confidence: {confidence:.3f}")
                    
                    # Save test result
                    os.makedirs('quick_test_results', exist_ok=True)
                    results[0].save(filename='quick_test_results/test_result.jpg')
                    print(f"✅ Test result saved: quick_test_results/test_result.jpg")
                else:
                    print("⚠️ No detections in test image")
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {model.names}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🚗 FAST License Plate Training - YOLOv8")
    print("=" * 50)
    print("⚡ Optimized for quick training and testing")
    
    # Verify dataset structure
    structure_ok, total_images = verify_dataset_structure()
    if not structure_ok:
        print("\n❌ Dataset structure issues found. Please fix before training.")
        return
    
    # Check data.yaml
    if not check_data_yaml():
        print("\n❌ data.yaml issues found. Please fix before training.")
        return
    
    # Optimize training parameters
    epochs, batch_size, imgsz, patience = optimize_training_params(total_images)
    
    # Ask user for confirmation
    print(f"\n⚡ FAST TRAINING MODE")
    print(f"   This will train for {epochs} epochs (instead of 100)")
    print(f"   Estimated time: {epochs * 0.5:.1f}-{epochs * 2:.1f} minutes")
    
    response = input("Continue with fast training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print("\n🚀 Starting FAST training process...")
    
    # Train model
    model_path = train_model_fast(epochs, batch_size, imgsz, patience)
    
    if model_path:
        print("\n🎉 FAST Training completed successfully!")
        print(f"✅ Model saved at: {model_path}")
        print(f"📁 Training artifacts: runs/detect/license_plate_detector_fast/")
        
        # Quick test
        if quick_test_model(model_path):
            print(f"\n✅ Model is working correctly!")
            print(f"\nNext steps:")
            print(f"1. Check test results in: quick_test_results/")
            print(f"2. Run full testing: python test_model.py")
            print(f"3. Start the web app: python main.py")
            print(f"4. If accuracy is low, train longer with more epochs")
        else:
            print(f"\n⚠️ Model test had issues, but training completed")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()