import glob
import os

root_dir = '/home/hiperwall/reid_poc_codes/archive/Market-1501-v15.09.15/query'

pattern = os.path.join(root_dir, "*.jpg")

obj_ids = set()
for image_path in sorted(glob.glob(pattern)):
    filename = os.path.basename(image_path)
    # Extract object_id (first substring before underscore)
    try:
        obj_id = int(filename.split("_")[0])
    except ValueError:
        print(f"[WARN] Cannot parse object_id from filename: {filename}")
        continue
    
    obj_ids.add(obj_id)

print(len(obj_ids))