import uuid
import faiss
import torch
import msgpack
import numpy as np
# import cv2  # removed cv2
import zmq
import torchreid
from threading import Thread
from fastapi import FastAPI
from torchvision import transforms
from PIL import Image
import threading
import io

# for time testing 
import time

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FAISS 셋업
dim = 512
threshold = 0.73
invalid_extract = 0

# Preprocessing ( transform to tensor )
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_osnet_feature(image_input, model, device=device):
    # if numpy array, convert directly to PIL (assume RGB)
    if isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input).convert('RGB')

    tensor = preprocess(img).unsqueeze(0).to(device)
    model.eval()
    
    # img.show()  # for debugging
    with torch.no_grad():
        feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu()

def assign_object_id(new_vector, threshold=threshold):
    global invalid_extract  # count False Negative.

    # 1. 입력 텐서를 numpy 배열로 변환하고 float32 타입, 1×D 형태로 reshape
    np_vec = new_vector.numpy().astype('float32').reshape(1, -1)

    # 2. L2 정규화: 벡터 길이가 1이 되도록
    faiss.normalize_L2(np_vec)

    # 3. Faiss 인덱스에서 가장 가까운 이웃 1개 검색
    D, I = index.search(np_vec, 1)

    # 4. 검색된 거리(또는 유사도) 값을 꺼내 float로 변환
    sim = float(D[0][0])

    # 5. 유사도 임계치 이상이면 기존 객체
    if sim >= threshold:
        return int(I[0][0])

    # 6. 새로운 객체로 판단될 때 고유 ID 생성
    new_id = uuid.uuid4().int & ((1<<63)-1)

    # 7. 생성한 ID와 함께 새 벡터를 인덱스에 추가
    index.add_with_ids(np_vec, np.array([new_id], dtype=np.int64))

    # 8. False Negative 카운터 증가
    invalid_extract += 1
    obj_list.append(new_id)

    # 9. 새로 만든 ID 반환
    return new_id

# dummy 저장 함수
def save_to_sqlite(obj_id, camera_id, timestamp):
    print(f"Saved: {obj_id}, cam: {camera_id}, time: {timestamp}")

# FAISS index 초기화
if device.type == 'cuda':
    res = faiss.StandardGpuResources()
    # CPU index 생성 후 IDMap으로 래핑
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIDMap(cpu_index)
    # GPU로 변환
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
else:
    # CPU에서 IDMap으로만 래핑된 IndexFlatIP 사용
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

# 모델 로드
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1,
    pretrained=True
).to(device)

obj_list = []
total_time = []

app = FastAPI()

# 디버그 헬퍼
def debug_image_data(img_payload, step_name=""):
    print(f"\n=== DEBUG {step_name} ===")
    print(f"Type: {type(img_payload)}")
    
    if hasattr(img_payload, '__len__'):
        print(f"Length: {len(img_payload)}")
    
    if isinstance(img_payload, (list, bytearray, bytes)):
        if len(img_payload) > 0:
            print(f"First 10 bytes: {img_payload[:10]}")
            print(f"Last 10 bytes: {img_payload[-10:]}")
    
    if isinstance(img_payload, np.ndarray):
        print(f"Shape: {img_payload.shape}")
        print(f"Dtype: {img_payload.dtype}")
        print(f"Min/Max values: {img_payload.min()}/{img_payload.max()}")


def zmq_worker():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind("tcp://*:5555")
    
    while True:
        try:
            msg = sock.recv()
            data = msgpack.unpackb(msg, raw=False)
        
            img_payload = data.get("image")
            if img_payload is None:
                print("fail to read img payload")
                continue

            # for time testing.
            start = time.perf_counter()
            
            # PIL로 디코딩 (RGB)
            img = Image.open(io.BytesIO(img_payload)).convert('RGB')
            img_array = np.array(img)
          
        except Exception as e:
            print(f"Fail to extract bytes {e}")
            continue
        
        try:
            feat = extract_osnet_feature(img_array, model)
            obj_id = assign_object_id(feat)
            

        except Exception as e:
            print(f"fail to decode img. {e}")
            continue
        
        # for time testing.
        end = time.perf_counter()
        total_time.append(end - start)
        print(f"run time: {end - start:.6f}sec")
        try:
            save_to_sqlite(obj_id, data["camera_id"], data["timestamp"])
        
        except Exception as e:
            print(f"fail to save to sqlite: {e}")
            continue

# 백그라운드 ZMQ 리스너 실행
Thread(target=zmq_worker, daemon=True).start()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/get_numofobjs")
def get_num_of_objs():
    print(len(obj_list))
    return {"status" : "ok"}

@app.get("/get_total_time")
def get_total_time():
    sum = 0
    for t in total_time:
        sum += t / len(total_time)
    print(f"total time : {sum:.6f}sec")
    return {"status" : "ok"}
# 실행 예: uvicorn main:app --host 0.0.0.0 --port 8000
