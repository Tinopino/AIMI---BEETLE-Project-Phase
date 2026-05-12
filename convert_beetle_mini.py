import os
import json
import glob
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

def create_mini_nnunet_dataset():
    wsi_dir = "data/images/development/wsis"
    mask_dir = "data/annotations/masks"
    output_dir = os.environ.get("nnUNet_raw", "./nnunet_raw")
    
    dataset_name = "Dataset301_BEETLE"
    out_base = os.path.join(output_dir, dataset_name)
    images_tr = os.path.join(out_base, "imagesTr")
    labels_tr = os.path.join(out_base, "labelsTr")
    
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {
            "background": 0,
            "other": 1,
            "non-invasive epithelium": 2,
            "invasive epithelium": 3,
            "necrosis": 4
        },
        "numTraining": 0, 
        "file_ending": ".png"
    }

    wsi_files = sorted(glob.glob(os.path.join(wsi_dir, "*.tif")))
    
    patch_size = 512
    stride = 512
    max_patches = 200
    patch_count = 0
    
    print(f"Extracting up to {max_patches} patches (size {patch_size}x{patch_size})...")

    for wsi_path in wsi_files:
        if patch_count >= max_patches:
            break
            
        filename = os.path.basename(wsi_path)
        case_id = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_dir, filename)
        
        if not os.path.exists(mask_path):
            continue
            
        try:
            image = tifffile.imread(wsi_path)
            mask = tifffile.imread(mask_path)
        except Exception as e:
            print(f"Skipping {case_id}: {e}")
            continue
            
        h, w = image.shape[:2]
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if patch_count >= max_patches:
                    break
                    
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Require at least 5% annotated tissue
                if np.sum(mask_patch > 0) / (patch_size * patch_size) > 0.05:
                    img_patch = image[y:y+patch_size, x:x+patch_size]
                    patch_id = f"{case_id}_{x}_{y}"
                    
                    # Save R, G, B channels as separate files (nnU-Net 2D requirement)
                    Image.fromarray(img_patch[:,:,0]).save(os.path.join(images_tr, f"{patch_id}_0000.png"))
                    Image.fromarray(img_patch[:,:,1]).save(os.path.join(images_tr, f"{patch_id}_0001.png"))
                    Image.fromarray(img_patch[:,:,2]).save(os.path.join(images_tr, f"{patch_id}_0002.png"))
                    
                    # Save mask
                    Image.fromarray(mask_patch.astype(np.uint8)).save(os.path.join(labels_tr, f"{patch_id}.png"))
                    
                    patch_count += 1

    dataset_json["numTraining"] = patch_count
    with open(os.path.join(out_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"Done. Saved {patch_count} patches to {out_base}.")

if __name__ == "__main__":
    create_mini_nnunet_dataset()