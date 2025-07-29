import os
import json
import base64
import io
import time
import uuid
import numpy as np
import torch
import faiss
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
from torchvision import transforms
from PIL import Image
import torchreid

# Load environment variables
load_dotenv('env/aws.env')
BROKER = os.getenv('BROKER')
REQUEST_TOPIC = os.getenv('REID_REQUEST_TOPIC')
RESPONSE_TOPIC = os.getenv('REID_RESPONSE_TOPIC')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FAISS index parameters
dim = 512
THRESHOLD = float(os.getenv('REID_THRESHOLD', 0.73))

# Initialize FAISS index
def init_faiss_index():
    if device.type == 'cuda':
        res = faiss.StandardGpuResources()
        cpu_idx = faiss.IndexFlatIP(dim)
        cpu_idx = faiss.IndexIDMap(cpu_idx)
        return faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    else:
        return faiss.IndexIDMap(faiss.IndexFlatIP(dim))

index = init_faiss_index()

# OSNet model loading
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=0,
    pretrained=True
).to(device)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(img: Image.Image) -> np.ndarray:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu().numpy().astype('float32').reshape(1, -1)

# Assign or retrieve global ID via FAISS
def assign_global_id(feat_vec: np.ndarray) -> int:
    # L2 normalize
    faiss.normalize_L2(feat_vec)
    # Search nearest
    D, I = index.search(feat_vec, 1)
    sim = float(D[0][0])
    if sim >= THRESHOLD:
        return int(I[0][0])
    # New ID
    new_id = uuid.uuid4().int & ((1<<63)-1)
    index.add_with_ids(feat_vec, np.array([new_id], dtype='int64'))
    return new_id

# Kafka setup\
consumer = KafkaConsumer(
    REQUEST_TOPIC,
    bootstrap_servers=[BROKER],
    value_deserializer=lambda b: json.loads(b.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='reid-global-service'
)
producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    value_serializer=lambda obj: json.dumps(obj).encode('utf-8')
)

print(f"[*] Central ReID service listening on topic '{REQUEST_TOPIC}' and publishing to '{RESPONSE_TOPIC}'")

# Main loop
for msg in consumer:
    data = msg.value
    cam_id = data.get('camera_id')
    local_id = data.get('local_id')
    ts = data.get('timestamp')
    b64 = data.get('crop_jpg', '')
    try:
        # Decode crop
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        print(f"[Error] Invalid crop image: {e}")
        continue
    # Extract feature and assign global ID
    start = time.perf_counter()
    feat = extract_feature(img)
    global_id = assign_global_id(feat)
    elapsed = time.perf_counter() - start
    # Prepare response
    response = {
        'camera_id': cam_id,
        'local_id': local_id,
        'timestamp': ts,
        'global_id': global_id,
        'elapsed': elapsed
    }
    # Send back
    producer.send(RESPONSE_TOPIC, response)
    # Optionally save to DB
    print(f"[ReID] cam={cam_id} local={local_id} -> global={global_id} ({elapsed:.3f}s)")
producer.flush()
