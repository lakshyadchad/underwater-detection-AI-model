import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
# Path to the autodistill labeled dataset (contains subdirectories with train/valid splits)
SOURCE_DIR = "" 

# Where to create the final consolidated YOLO dataset
OUTPUT_DIR = ""
SPLIT_RATIO = 0.8 # 80% Train, 20% Val

def create_dirs():
    """Creates the empty folder structure."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def assemble_dataset():
    create_dirs()
    
    # Get all subdirectories (each class category)
    subdirs = [d for d in os.listdir(SOURCE_DIR) 
               if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found {len(subdirs)} categories: {subdirs}")
    
    total_train_imgs = 0
    total_val_imgs = 0
    
    # Process each category subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(SOURCE_DIR, subdir)
        
        # Check if autodistill created train/valid folders
        train_img_path = os.path.join(subdir_path, "train", "images")
        train_lbl_path = os.path.join(subdir_path, "train", "labels")
        valid_img_path = os.path.join(subdir_path, "valid", "images")
        valid_lbl_path = os.path.join(subdir_path, "valid", "labels")
        
        # Copy training data
        if os.path.exists(train_img_path):
            for img_file in os.listdir(train_img_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    src_img = os.path.join(train_img_path, img_file)
                    dst_img = os.path.join(OUTPUT_DIR, "images", "train", f"{subdir}_{img_file}")
                    shutil.copy(src_img, dst_img)
                    
                    # Copy corresponding label
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    src_lbl = os.path.join(train_lbl_path, label_file)
                    if os.path.exists(src_lbl):
                        dst_lbl = os.path.join(OUTPUT_DIR, "labels", "train", f"{subdir}_{label_file}")
                        shutil.copy(src_lbl, dst_lbl)
                    
                    total_train_imgs += 1
        
        # Copy validation data
        if os.path.exists(valid_img_path):
            for img_file in os.listdir(valid_img_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    src_img = os.path.join(valid_img_path, img_file)
                    dst_img = os.path.join(OUTPUT_DIR, "images", "val", f"{subdir}_{img_file}")
                    shutil.copy(src_img, dst_img)
                    
                    # Copy corresponding label
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    src_lbl = os.path.join(valid_lbl_path, label_file)
                    if os.path.exists(src_lbl):
                        dst_lbl = os.path.join(OUTPUT_DIR, "labels", "val", f"{subdir}_{label_file}")
                        shutil.copy(src_lbl, dst_lbl)
                    
                    total_val_imgs += 1
        
        print(f"✓ Processed '{subdir}'")
    
    print(f"\nTotal Images:")
    print(f"Training: {total_train_imgs} | Validation: {total_val_imgs}")
    print(f"Done! Consolidated dataset saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    assemble_dataset()

    