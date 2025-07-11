# send_test.py
import zmq
import cv2
import msgpack
import time
import glob
import os

# ZeroMQ setup
ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://localhost:5555")

# Directory containing jpg images
root_dir = '/home/hiperwall/reid_poc_codes/archive/Market-1501-v15.09.15/query'
camera_id = "cam01"

# Iterate over all jpg files in the directory
pattern = os.path.join(root_dir, "*.jpg")
for image_path in sorted(glob.glob(pattern)):
    filename = os.path.basename(image_path)
    # Extract object_id (first substring before underscore)
    try:
        obj_id = int(filename.split("_")[0])
    except ValueError:
        print(f"[WARN] Cannot parse object_id from filename: {filename}")
        continue

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {filename}")
        continue

    # Encode image to JPEG buffer
    retval, buf = cv2.imencode(".jpg", img)
    if not retval:
        print(f"[ERROR] Failed to encode image: {filename}")
        continue

    # Prepare payload with image bytes and metadata
    meta = {
        "object_id": obj_id,
        "camera_id": camera_id,
        "timestamp": time.time()
    }
    payload = {"image": buf.tobytes(), **meta}

    # Send message
    sock.send(msgpack.packb(payload, use_bin_type=True))
    print(f"Sent {filename} (object_id={obj_id})")

    # Optional: small delay
    time.sleep(0.01)
