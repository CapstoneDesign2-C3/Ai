from kafka import KafkaProducer
from dotenv import load_dotenv
import json
import cv2
import os
import base64
import numpy as np
import time
from typing import Optional, Dict, Any, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseProducer:
    """Kafka Producer 기본 클래스"""
    
    def __init__(self, topic_env_key: str, 
                 key_serializer=None, 
                 value_serializer=None,
                 acks: int = 0,
                 retries: int = 3,
                 batch_size: int = 16384,
                 linger_ms: int = 10):
        """
        기본 Producer 초기화
        
        Args:
            topic_env_key: 환경변수에서 읽을 토픽 키
            key_serializer: 키 직렬화 함수
            value_serializer: 값 직렬화 함수
            acks: 확인 레벨
            retries: 재시도 횟수
            batch_size: 배치 크기
            linger_ms: 배치 대기 시간
        """
        # 환경변수 로드
        dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
        load_dotenv(dotenv_path)
        
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('BASIC')
        
        if not self.broker or not self.topic:
            raise ValueError(f"BROKER 또는 {topic_env_key} 환경변수가 설정되지 않았습니다.")
        
        # Producer 설정
        producer_config = {
            'bootstrap_servers': self.broker,
            'acks': acks,
            'api_version': (2, 5, 0),
            'retries': retries,
            'batch_size': batch_size,
            'linger_ms': linger_ms,  # 성능 향상을 위한 배치 처리
        }
        
        if key_serializer:
            producer_config['key_serializer'] = key_serializer
        if value_serializer:
            producer_config['value_serializer'] = value_serializer
            
        self.producer = KafkaProducer(**producer_config)
        self._message_count = 0
        self._flush_interval = 10  # 10개 메시지마다 flush
        
        logger.info(f"✅ {self.__class__.__name__} 초기화 완료 - Topic: {self.topic}")
    
    def _should_flush(self) -> bool:
        """flush 여부 결정"""
        self._message_count += 1
        return self._message_count % self._flush_interval == 0
    
    def _send_with_callback(self, topic: str, key: Optional[bytes], value: bytes) -> Dict[str, Any]:
        """콜백과 함께 메시지 전송"""
        try:
            future = self.producer.send(topic, key=key, value=value)
            
            # 성능 최적화: 매번 flush하지 않음
            if self._should_flush():
                self.producer.flush()
            
            return {'status_code': 200, 'error': None}
            
        except Exception as e:
            logger.error(f"❌ 메시지 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}
    
    def close(self):
        """Producer 종료"""
        try:
            self.producer.flush()
            self.producer.close()
            logger.info(f"✅ {self.__class__.__name__} 종료 완료")
        except Exception as e:
            logger.error(f"❌ Producer 종료 중 오류: {e}")


class FrameProducer(BaseProducer):
    """프레임 전송용 Producer"""
    
    def __init__(self, camera_id: str):
        super().__init__(
            topic_env_key='FRAME_TOPIC',
            key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
            value_serializer=lambda x: x  # 바이너리 데이터 그대로
        )
        self.camera_id = str(camera_id)
        
    def send_message(self, frame: np.ndarray, 
                    quality: int = 90, 
                    format: str = '.jpg') -> Dict[str, Any]:
        """
        프레임 메시지 전송
        
        Args:
            frame: 프레임 이미지 (numpy array)
            quality: JPEG 압축 품질 (1-100)
            format: 이미지 포맷 ('.jpg', '.png' 등)
        """
        try:
            # 프레임 인코딩
            if format == '.jpg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            else:
                encode_params = []
                
            success, buffer = cv2.imencode(format, frame, encode_params)
            
            if not success:
                raise ValueError("프레임 인코딩 실패")
            
            jpeg_bytes = buffer.tobytes()
            
            # 메타데이터 추가 (선택사항)
            timestamp = int(time.time() * 1000)  # milliseconds
            
            return self._send_with_callback(
                self.topic,
                key=self.camera_id.encode('utf-8'),
                value=jpeg_bytes
            )
            
        except Exception as e:
            logger.error(f"❌ 프레임 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}


class DetectedResultProducer(BaseProducer):
    """검출 결과 전송용 Producer"""
    
    def __init__(self):
        super().__init__(
            topic_env_key='DETECTED_RESULT',
            key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
            value_serializer=lambda x: json.dumps(x).encode('utf-8') if isinstance(x, (dict, list)) else x
        )
    
    def send_message(self, camera_id: str, payload: Union[Dict, str, bytes]) -> Dict[str, Any]:
        """
        검출 결과 메시지 전송
        
        Args:
            camera_id: 카메라 ID
            payload: 검출 결과 데이터 (dict, str, bytes)
        """
        try:
            # payload 타입에 따른 처리
            if isinstance(payload, dict):
                # 타임스탬프 추가
                payload['timestamp'] = int(time.time() * 1000)
                value = json.dumps(payload).encode('utf-8')
            elif isinstance(payload, str):
                value = payload.encode('utf-8')
            elif isinstance(payload, bytes):
                value = payload
            else:
                raise ValueError(f"지원하지 않는 payload 타입: {type(payload)}")
            
            return self._send_with_callback(
                self.topic,
                key=camera_id.encode('utf-8'),
                value=value
            )
            
        except Exception as e:
            logger.error(f"❌ 검출 결과 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}


class TrackResultProducer(BaseProducer):
    """추적 결과(Crop) 전송용 Producer"""
    
    def __init__(self, camera_id: str):
        super().__init__(
            topic_env_key='REID_REQUEST',
            key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
            value_serializer=lambda x: json.dumps(x).encode('utf-8') if isinstance(x, dict) else x
        )
        self.camera_id = str(camera_id)
    
    def send_message(self, crop: np.ndarray, 
                    track_id: Optional[int] = None,
                    bbox: Optional[list] = None,
                    confidence: Optional[float] = None,
                    class_name: Optional[str] = None,
                    encoding: str = 'base64') -> Dict[str, Any]:
        """
        추적 결과 메시지 전송
        
        Args:
            crop: 크롭된 이미지 (numpy array)
            track_id: 추적 ID
            bbox: 바운딩 박스 [x, y, w, h]
            confidence: 신뢰도
            class_name: 클래스 이름
            encoding: 인코딩 방식 ('base64' 또는 'binary')
        """
        try:
            if encoding == 'base64':
                # Base64 인코딩 방식
                return self._send_base64_message(crop, track_id, bbox, confidence, class_name)
            elif encoding == 'binary':
                # 바이너리 방식
                return self._send_binary_message(crop, track_id, bbox, confidence, class_name)
            else:
                raise ValueError(f"지원하지 않는 인코딩 방식: {encoding}")
                
        except Exception as e:
            logger.error(f"❌ 추적 결과 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}
    
    def _send_base64_message(self, crop: np.ndarray, track_id: Optional[int], 
                           bbox: Optional[list], confidence: Optional[float], 
                           class_name: Optional[str]) -> Dict[str, Any]:
        """Base64 인코딩으로 메시지 전송"""
        # 이미지를 JPEG로 인코딩
        success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("크롭 이미지 인코딩 실패")
        
        # Base64 인코딩
        crop_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # 메타데이터와 함께 JSON 구성
        message = {
            'camera_id': self.camera_id,
            'timestamp': int(time.time() * 1000),
            'crop_image': crop_b64,
            'image_format': 'jpeg',
            'encoding': 'base64'
        }
        
        # 선택적 필드 추가
        if track_id is not None:
            message['track_id'] = track_id
        if bbox is not None:
            message['bbox'] = bbox
        if confidence is not None:
            message['confidence'] = confidence
        if class_name is not None:
            message['class_name'] = class_name
        
        value = json.dumps(message).encode('utf-8')
        
        return self._send_with_callback(
            self.topic,
            key=self.camera_id.encode('utf-8'),
            value=value
        )
    
    def _send_binary_message(self, crop: np.ndarray, track_id: Optional[int],
                           bbox: Optional[list], confidence: Optional[float],
                           class_name: Optional[str]) -> Dict[str, Any]:
        """바이너리 방식으로 메시지 전송"""
        # 이미지를 JPEG로 인코딩
        success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("크롭 이미지 인코딩 실패")
        
        jpeg_bytes = buffer.tobytes()
        
        # 헤더 정보를 별도로 전송하거나 바이너리에 포함할 수 있음
        # 여기서는 단순히 이미지만 전송
        return self._send_with_callback(
            self.topic,
            key=self.camera_id.encode('utf-8'),
            value=jpeg_bytes
        )


# Factory 함수들
def create_frame_producer(camera_id: str) -> FrameProducer:
    """FrameProducer 생성"""
    return FrameProducer(camera_id)

def create_detected_result_producer() -> DetectedResultProducer:
    """DetectedResultProducer 생성"""
    return DetectedResultProducer()

def create_track_result_producer(camera_id: str) -> TrackResultProducer:
    """TrackResultProducer 생성"""
    return TrackResultProducer(camera_id)


# 사용 예시
if __name__ == "__main__":
    # 테스트용 코드
    import numpy as np
    
    # 더미 이미지 생성
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        # Producer 생성
        frame_producer = create_frame_producer("camera_001")
        result_producer = create_detected_result_producer()
        track_producer = create_track_result_producer("camera_001")
        
        # 메시지 전송 테스트
        print("📤 프레임 전송 테스트")
        result1 = frame_producer.send_message(test_frame)
        print(f"결과: {result1}")
        
        print("📤 검출 결과 전송 테스트")
        detection_result = {
            "detections": [
                {"class": "person", "confidence": 0.95, "bbox": [100, 200, 50, 100]}
            ]
        }
        result2 = result_producer.send_message("camera_001", detection_result)
        print(f"결과: {result2}")
        
        print("📤 추적 결과 전송 테스트")
        result3 = track_producer.send_message(
            test_crop, 
            track_id=123, 
            bbox=[100, 200, 50, 100],
            confidence=0.95,
            class_name="person"
        )
        print(f"결과: {result3}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    finally:
        # Producer 종료
        frame_producer.close()
        result_producer.close()
        track_producer.close()