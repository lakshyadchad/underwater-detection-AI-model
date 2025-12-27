import os
import torch
from ultralytics import YOLO  # type: ignore

# --- CONFIGURATION ---
# Ensure this matches the file you created in Phase 3, Step 1
DATA_YAML = "" 
MODEL_TYPE = "yolo11n.pt"  # Nano model (Fastest for Edge/Mobile)
PROJECT_NAME = "maritime_security_project"
RUN_NAME = "underwater_yolo_v11"

def train_yolo():
    print(f"--- 🕵️ Starting Detector Training ---")

    # 1. Hardware Check
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not found. Training on CPU will be very slow!")
        device = 'cpu'
    else:
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        device = 0

    # 2. File Safety Check
    if not os.path.exists(DATA_YAML):
        print(f"❌ ERROR: '{DATA_YAML}' not found!")
        print("Please ensure you created dataset.yaml in this folder.")
        return

    # 3. Load Model
    # YOLOv11n (Nano) is the best balance of speed/accuracy for this problem
    print(f"Loading {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE) 

    # 4. Train
    print("🚀 Starting Training Loop...")
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=50,               # 50 is sufficient for Transfer Learning
            imgsz=640,               # Standard resolution
            batch=16,                # Reduce to 8 if you get Memory Errors
            device=device,
            
            # --- FILES & LOGGING ---
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True,           # Overwrite previous run if exists
            
            # --- PHYSICS-BASED AUGMENTATIONS (CRITICAL) ---
            # Default YOLO flips images upside down. We MUST disable this.
            flipud=0.0,              # Submarines/Mines don't float upside down
            fliplr=0.5,              # Left/Right flip is okay (Sub can face left or right)
            degrees=10.0,            # Slight rotation (rolling camera) is okay
            
            # --- OPTIMIZATIONS ---
            workers=0,               # 0 = No multiprocessing (fixes Windows errors)
            optimizer='auto',        # Ultralytics chooses best (usually AdamW)
            verbose=True
        )
        print("✅ Training Complete!")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ ERROR: GPU Out of Memory.")
            print("SOLUTION: Open the script and change 'batch=16' to 'batch=8' or 'batch=4'.")
        else:
            raise e

    # 5. Validation
    print("📊 Validating Best Model...")
    metrics = model.val()
    print(f"Final mAP50: {metrics.box.map50:.4f}")

    # 6. Export for Deployment (ONNX & TFLite)
    print("📦 Exporting for Edge Deployment...")
    # Export ONNX (For Jetson/PC)
    model.export(format='onnx', dynamic=True, simplify=True)
    # Export TFLite (For Android Smartphone)
    # Note: TFLite export might take a few minutes
    try:
        model.export(format='tflite', int8=False, half=True)
        print("✅ Models exported to 'runs/detect/underwater_yolo_v11/weights/'")
    except Exception as e:
        print(f"⚠️ TFLite Export Warning (You can ignore if not using Android): {e}")

if __name__ == "__main__":
    train_yolo()