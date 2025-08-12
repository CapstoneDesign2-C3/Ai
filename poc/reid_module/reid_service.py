import os
import sys
import json
import base64
import io
import time
import uuid
import logging
from typing import Optional

import numpy as np
import torch
import faiss
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
from PIL import Image
import torchreid
from torchvision import transforms


from db_util.db_util import PostgreSQL


class ReIDService:
    def __init__(self, db: PostgreSQL):
        # --- Env ---
        load_dotenv('/home/hiperwall/Ai_modules/Ai/env/aws.env')
        self.broker = os.getenv('BROKER')
        self.request_topic = os.getenv('REID_REQUEST_TOPIC')
        self.response_topic = os.getenv('REID_RESPONSE_TOPIC')  # 중요: 응답 토픽

        if not self.broker or not self.request_topic or not self.response_topic:
            raise RuntimeError("BROKER/REID_REQUEST_TOPIC/REID_RESPONSE_TOPIC env가 필요합니다.")

        # --- DB ---
        self.db = db

        # --- Device / Torch ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # --- ReID 설정 ---
        self.dim = 512
        self.threshold = float(os.getenv('REID_THRESHOLD', 0.73))

        # --- Init ---
        self._init_faiss_index()
        self._init_model()
        self._init_kafka()
        self._init_preprocessing()

        # --- Logger ---
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ReIDService")
        self.logger.info(f"ReID service initialized on device: {self.device}")
        self.logger.info(f"Listening on '{self.request_topic}', publishing to '{self.response_topic}'")

    # ----------------------- Init -----------------------
    def _init_faiss_index(self):
        """FAISS index (IP + IDMap). GPU 가능 시 GPU index 사용."""
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            cpu_idx = faiss.IndexFlatIP(self.dim)
            cpu_idx = faiss.IndexIDMap(cpu_idx)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def _init_model(self):
        """OSNet feature extractor 로드."""
        # osnet_x1_0은 512-d 임베딩을 출력
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=0,
            pretrained=True
        ).to(self.device)
        self.model.eval()

        # (선택) 워밍업: 빈 텐서 한 번 태워서 첫 추론 지연 줄이기
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 128, device=self.device)
            _ = self.model(dummy)

    def _init_kafka(self):
        """Kafka consumer/producer 초기화."""
        self.consumer = KafkaConsumer(
            self.request_topic,
            bootstrap_servers=[self.broker],
            value_deserializer=lambda b: json.loads(b.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='reid-global-service'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda obj: json.dumps(obj).encode('utf-8')
        )

    def _init_preprocessing(self):
        """이미지 전처리 파이프라인 (OSNet 표준)."""
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ----------------------- Core -----------------------
    def extract_feature(self, img: Image.Image) -> np.ndarray:
        """PIL.Image -> 1x512 float32 (L2 normalized)."""
        try:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)              # (1,512)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).detach().cpu().numpy().astype('float32').reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise

    def assign_global_id(self, feat_vec: np.ndarray, img_bytes: bytes, code_name: str = "사람") -> tuple[int, bool]:
        """
        FAISS로 최근접 검색 후 임계치 미만이면 신규 global_id 생성.
        신규일 때 detected_object 테이블에 썸네일/feature 저장.
        """
        try:
            faiss.normalize_L2(feat_vec)
            D, I = self.index.search(feat_vec, 1)   # D: similarity or distance | I: index of similarity 

            if len(I[0]) > 0 and I[0][0] != -1: # search result is valid
                sim = float(D[0][0])

                # already exist object.
                if sim >= self.threshold:
                    # TODO: add crop image to detection / need index -> uuid method
                    self.db.addNewDetection()
                    return int(I[0][0]), True

            # 신규 등록
            new_id = uuid.uuid4().int & ((1 << 63) - 1)
            self.index.add_with_ids(feat_vec, np.array([new_id], dtype='int64'))

            # feature 직렬화(base64 of float32 bytes)
            feat_b64 = base64.b64encode(feat_vec.astype('float32').tobytes()).decode('utf-8')

            # DB 저장 (BYTEA에 JPEG bytes)
            self.db.addNewDetectedObject(uuid=new_id, crop_img=img_bytes, feature=feat_b64, code_name=code_name)
            return new_id, False

        except Exception as e:
            self.logger.error(f"Global ID assignment failed: {e}")
            raise

    def process_reid_request(self, data: dict) -> dict:
        """Kafka 메시지 1건 처리 → global_id 결정 → 응답 payload 생성."""
        camera_id = data.get('camera_id')
        local_id  = data.get('local_id') or data.get('track_id')
        image_base64 = data.get('crop_jpg', '')
        class_name = data.get('class_name', 'person')  # 한글 '사람' 코드와 매핑 목적

        # crop decode
        try:
            img_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            self.logger.error(f"Invalid crop image from cam={camera_id} : {e}")
            raise

        # feature & id 결정
        start = time.perf_counter()
        feature_vector = self.extract_feature(img)

        code_name = "사람" if class_name.lower() == "person" else class_name
        global_id, is_exist = self.assign_global_id(feature_vector, img_bytes, code_name=code_name)
        elapsed = time.perf_counter() - start

        now_ms = int(time.time() * 1000)
        detection_id = self.db.addNewDetection(
            uuid=global_id,
            appeared_time=now_ms,
            exit_time=now_ms        # 임시값, tracker에서 퇴장시 update
        )


        # 응답(로컬 매칭 위해 local_id 포함)
        response = {
        "camera_id": camera_id,
        "local_id": local_id,
        "global_id": global_id,
        "detection_id": detection_id,   # ← tracker가 보관
        "appeared_time": now_ms,        # ← 필요 시 자연키 대체
        "existing": bool(is_exist),
        "status": "success"
        }

        self.logger.info(f"[ReID] cam={camera_id} local={local_id} -> global={global_id} ({elapsed*1000:.1f} ms)")
        return response

    def send_response(self, response: dict):
        """Kafka 응답 publish."""
        try:
            self.producer.send(self.response_topic, response)
            self.producer.flush()
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
            raise

    # ----------------------- Loop -----------------------
    def run(self):
        self.logger.info("Starting ReID service...")
        try:
            for msg in self.consumer:
                data = msg.value
                try:
                    response = self.process_reid_request(data)
                    self.send_response(response)
                except Exception as e:
                    # 실패 응답도 회신(로컬 매핑/리트라이 판단용)
                    error_response = {
                        'camera_id': data.get('camera_id', 'unknown'),
                        'local_id': data.get('local_id', data.get('track_id', -1)),
                        'timestamp': data.get('timestamp', time.time()),
                        'global_id': -1,
                        'elapsed': 0.0,
                        'status': 'error',
                        'error': str(e)
                    }
                    self.send_response(error_response)
        except KeyboardInterrupt:
            self.logger.info("Shutting down ReID service...")
        finally:
            self.cleanup()

    def cleanup(self):
        if hasattr(self, 'consumer'):
            try: self.consumer.close()
            except Exception: pass
        if hasattr(self, 'producer'):
            try: self.producer.close()
            except Exception: pass
        self.logger.info("ReID service shutdown complete")


if __name__ == "__main__":
    # .env에서 DB 연결정보 읽어 DB 핸들 생성 후 서비스 기동
    load_dotenv('/home/hiperwall/Ai_modules/Ai/env/aws.env')
    db = PostgreSQL(
        os.getenv('DB_HOST'), os.getenv('DB_NAME'),
        os.getenv('DB_USER'), os.getenv('DB_PASSWORD'),
        os.getenv('DB_PORT')
    )
    ReIDService(db=db).run()
