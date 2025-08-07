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
from poc.db_util.db_util import PostgreSQL
from PIL import Image
import torchreid
import logging

class ReIDService:
    def __init__(self, db: PostgreSQL):
        # Load environment variables
        load_dotenv('env/aws.env')
        self.broker = os.getenv('BROKER')
        self.request_topic = os.getenv('REID_REQUEST_TOPIC')
        
        #DB
        self.db = db

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # FAISS index parameters
        self.dim = 512
        self.threshold = float(os.getenv('REID_THRESHOLD', 0.73))
        
        # Initialize components
        self._init_faiss_index()
        self._init_model()
        self._init_kafka()
        self._init_preprocessing()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ReID service initialized on device: {self.device}")
        self.logger.info(f"Listening on topic '{self.request_topic}', publishing to '{self.response_topic}'")
    
    def _init_faiss_index(self):
        """Initialize FAISS index for feature matching"""
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources()
            cpu_idx = faiss.IndexFlatIP(self.dim)
            cpu_idx = faiss.IndexIDMap(cpu_idx)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
    
    def _init_model(self):
        """Initialize OSNet model for feature extraction"""
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=0,
            pretrained=True
        ).to(self.device)
        self.model.eval()
    
    def _init_kafka(self):
        """Initialize Kafka consumer and producer"""
        self.consumer = KafkaConsumer(
            self.request_topic,
            bootstrap_servers=[self.broker],
            value_deserializer=lambda b: json.loads(b.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='reid-global-service'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda obj: json.dumps(obj).encode('utf-8')
        )
    
    def _init_preprocessing(self):
        """Initialize image preprocessing pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_feature(self, img: Image.Image) -> np.ndarray:
        """Extract feature vector from person crop image"""
        try:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy().astype('float32').reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise
    
    def assign_global_id(self, feat_vec: np.ndarray, img) -> int:
        """Assign global ID using FAISS similarity search"""
        try:
            # L2 normalize feature vector
            faiss.normalize_L2(feat_vec)
            
            # Search for nearest neighbor
            D, I = self.index.search(feat_vec, 1)
            
            if len(I[0]) > 0 and I[0][0] != -1:
                sim = float(D[0][0])
                if sim >= self.threshold:
                    return int(I[0][0])
            
            # Create new global ID for new person
            new_id = uuid.uuid4().int & ((1<<63)-1)
            self.index.add_with_ids(feat_vec, np.array([new_id], dtype='int64'))
            # TODO feature 저장 방법 고민
            self.db.addNewDetectedObject(uuid=new_id, crop_img=img)
            return new_id
            
        except Exception as e:
            self.logger.error(f"Global ID assignment failed: {e}")
            raise
    
    def process_reid_request(self, data: dict) -> dict:
        """Process single ReID request"""
        camera_id = data.get('camera_id')
        image_base64 = data.get('crop_jpg', '')
        
        try:
            # Decode crop image
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        except Exception as e:
            self.logger.error(f"Invalid crop image from cam={camera_id} : {e}")
            raise
        
        # Extract feature and assign global ID
        start = time.perf_counter()
        feature_vector = self.extract_feature(img)
        global_id = self.assign_global_id(feature_vector, img)
        elapsed = time.perf_counter() - start
        
        # TODO addNewDetection 메소드 통해 db에 동선 저장, db.addNewDetection(self, uuid, appeared_time, exit_time)

        # Prepare response
        response = {
            'camera_id': camera_id,
            'global_id': global_id
        }

        self.logger.info(f"[ReID] cam={camera_id}  -> global={global_id} ({elapsed:.3f}s)")
        return response
    
    def send_response(self, response: dict):
        """Send response back through Kafka"""
        try:
            self.producer.send(self.response_topic, response)
            self.producer.flush()
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
            raise
    
    def run(self):
        """Main service loop"""
        self.logger.info("Starting ReID service...")
        
        try:
            for msg in self.consumer:
                try:
                    data = msg.value
                    response = self.process_reid_request(data)
                    self.send_response(response)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    # Send error response
                    error_response = {
                        'camera_id': data.get('camera_id', 'unknown'),
                        'local_id': data.get('local_id', 'unknown'),
                        'timestamp': data.get('timestamp', time.time()),
                        'global_id': -1,
                        'elapsed': 0,
                        'status': 'error',
                        'error': str(e)
                    }
                    self.send_response(error_response)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down ReID service...")
        except Exception as e:
            self.logger.error(f"Fatal error in ReID service: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'consumer'):
            self.consumer.close()
        if hasattr(self, 'producer'):
            self.producer.close()
        self.logger.info("ReID service shutdown complete")

if __name__ == "__main__":
    dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
    load_dotenv(dotenv_path)
    db = PostgreSQL(os.getenv('DB_HOST'), os.getenv('DB_NAME'), os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_PORT'))
    service = ReIDService(db=db)
    service.run()
