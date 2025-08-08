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
    """Kafka Producer 기본 클래스 - 파티션 지원"""
    
    def __init__(self, topic_env_key: str, 
                 key_serializer=None, 
                 value_serializer=None,
                 partitioner=None,
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
            partitioner: 파티셔너 (카메라 ID 기반)
            acks: 확인 레벨
            retries: 재시도 횟수
            batch_size: 배치 크기
            linger_ms: 배치 대기 시간
        """
        # 환경변수 로드
        dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
        load_dotenv(dotenv_path)
        
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv(topic_env_key)
        
        if not self.broker or not self.topic:
            raise ValueError(f"BROKER 또는 {topic_env_key} 환경변수가 설정되지 않았습니다.")
        
        # Producer 설정
        producer_config = {
            'bootstrap_servers': self.broker,
            'acks': acks,
            'api_version': (2, 5, 0),
            'retries': retries,
            'batch_size': batch_size,
            'linger_ms': linger_ms,
            'partitioner': partitioner # 카메라 ID 기반 파티션
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
    
    def _send_with_callback(self, topic: str, key: Optional[bytes], value: bytes, partition: Optional[int] = None) -> Dict[str, Any]:
        """콜백과 함께 메시지 전송"""
        try:
            future = self.producer.send(topic, key=key, value=value, partition=partition)
            
            # 성능 최적화: 매번 flush하지 않음
            if self._should_flush():
                self.producer.flush()
            
            return {'status_code': 200, 'error': None, 'partition': partition}
            
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
    """프레임 전송용 Producer - 카메라 ID로 파티션 분할"""
    
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
        프레임 메시지 전송 - 카메라 ID를 key로 파티션 결정
        
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
            
            # 카메라 ID를 key로 사용하여 파티션 자동 결정
            return self._send_with_callback(
                self.topic,
                key=self.camera_id.encode('utf-8'),
                value=jpeg_bytes
            )
            
        except Exception as e:
            logger.error(f"❌ 프레임 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}


class TrackResultProducer(BaseProducer):
    """추적 결과(Crop) 전송용 Producer - 카메라 ID로 파티션 분할하여 ReID 서비스로 전송"""
    
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
        추적 결과 메시지 전송 - 카메라 ID로 파티션 결정
        
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
                return self._send_base64_message(crop, track_id, bbox, confidence, class_name)
            elif encoding == 'binary':
                return self._send_binary_message(crop, track_id, bbox, confidence, class_name)
            else:
                raise ValueError(f"지원하지 않는 인코딩 방식: {encoding}")
                
        except Exception as e:
            logger.error(f"❌ 추적 결과 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}
    
    def send_crop_from_base64(self, crop_base64: str,
                             track_id: Optional[int] = None,
                             bbox: Optional[list] = None,
                             confidence: Optional[float] = None,
                             class_name: Optional[str] = None,
                             local_id: Optional[int] = None) -> Dict[str, Any]:
        """
        DetectorAndTracker.extract_crop_from_frame() 결과(base64)를 ReID 서비스로 전송
        기존 ReID 서비스 포맷에 맞춤
        
        Args:
            crop_base64: DetectorAndTracker에서 생성된 base64 문자열
            track_id: 추적 ID
            bbox: 바운딩 박스 [x, y, w, h]
            confidence: 신뢰도
            class_name: 클래스 이름
            local_id: 로컬 ID (track_id와 동일하게 사용)
        """
        try:
            # ReID 서비스 포맷에 맞춘 메시지 구성
            message = {
                'camera_id': self.camera_id,
                'crop_jpg': crop_base64,  # ReID 서비스에서 기대하는 필드명
                'timestamp': int(time.time() * 1000),
                'image_format': 'jpeg',
                'encoding': 'base64'
            }
            
            # 선택적 필드 추가
            if track_id is not None:
                message['track_id'] = track_id
                message['local_id'] = local_id if local_id is not None else track_id
            if bbox is not None:
                message['bbox'] = bbox
            if confidence is not None:
                message['confidence'] = confidence
            if class_name is not None:
                message['class_name'] = class_name
            
            value = json.dumps(message).encode('utf-8')
            
            # 카메라 ID를 key로 사용하여 파티션 자동 결정
            return self._send_with_callback(
                self.topic,
                key=self.camera_id.encode('utf-8'),
                value=value
            )
            
        except Exception as e:
            logger.error(f"❌ Base64 크롭 전송 실패: {e}")
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
        
        # ReID 서비스 포맷으로 메시지 구성
        return self.send_crop_from_base64(crop_b64, track_id, bbox, confidence, class_name)
    
    def _send_binary_message(self, crop: np.ndarray, track_id: Optional[int],
                           bbox: Optional[list], confidence: Optional[float],
                           class_name: Optional[str]) -> Dict[str, Any]:
        """바이너리 방식으로 메시지 전송"""
        # 이미지를 JPEG로 인코딩
        success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("크롭 이미지 인코딩 실패")
        
        jpeg_bytes = buffer.tobytes()
        
        # 카메라 ID를 key로 사용
        return self._send_with_callback(
            self.topic,
            key=self.camera_id.encode('utf-8'),
            value=jpeg_bytes
        )


class ReIDResponseProducer(BaseProducer):
    """ReID 응답 전송용 Producer - 카메라별로 응답 전달"""
    
    def __init__(self):
        super().__init__(
            topic_env_key='REID_RESPONSE',
            key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
            value_serializer=lambda x: json.dumps(x).encode('utf-8') if isinstance(x, dict) else x
        )
    
    def send_response(self, camera_id: str, global_id: int, 
                     local_id: Optional[int] = None,
                     track_id: Optional[int] = None,
                     elapsed: Optional[float] = None,
                     status: str = 'success') -> Dict[str, Any]:
        """
        ReID 응답 전송
        
        Args:
            camera_id: 카메라 ID
            global_id: 글로벌 ID
            local_id: 로컬 ID
            track_id: 추적 ID
            elapsed: 처리 시간
            status: 처리 상태
        """
        try:
            message = {
                'camera_id': camera_id,
                'global_id': global_id,
                'timestamp': int(time.time() * 1000),
                'status': status
            }
            
            # 선택적 필드 추가
            if local_id is not None:
                message['local_id'] = local_id
            if track_id is not None:
                message['track_id'] = track_id
            if elapsed is not None:
                message['elapsed'] = elapsed
            
            value = json.dumps(message).encode('utf-8')
            
            # 카메라 ID를 key로 사용하여 해당 카메라로 응답 전달
            return self._send_with_callback(
                self.topic,
                key=camera_id.encode('utf-8'),
                value=value
            )
            
        except Exception as e:
            logger.error(f"❌ ReID 응답 전송 실패: {e}")
            return {'status_code': 500, 'error': str(e)}


# Factory 함수들 - 파티션 기반
def create_frame_producer(camera_id: str) -> FrameProducer:
    """FrameProducer 생성"""
    return FrameProducer(camera_id)

def create_track_result_producer(camera_id: str) -> TrackResultProducer:
    """TrackResultProducer 생성"""
    return TrackResultProducer(camera_id)

def create_reid_response_producer() -> ReIDResponseProducer:
    """ReIDResponseProducer 생성"""
    return ReIDResponseProducer()


# 파티션 기반 테스트 코드
if __name__ == "__main__":
    import numpy as np
    
    # 여러 카메라 시뮬레이션
    camera_ids = ["camera_001", "camera_002", "camera_003"]
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        # 각 카메라별 Producer 생성
        producers = {}
        for camera_id in camera_ids:
            producers[camera_id] = {
                'frame': create_frame_producer(camera_id),
                'track': create_track_result_producer(camera_id)
            }
        
        reid_response_producer = create_reid_response_producer()
        
        # 파티션별 메시지 전송 테스트
        for i, camera_id in enumerate(camera_ids):
            print(f"📤 Camera {camera_id} 메시지 전송 테스트")
            
            # 프레임 전송
            result1 = producers[camera_id]['frame'].send_message(test_frame)
            print(f"  프레임 전송 결과: {result1}")
            
            # 추적 결과 전송 (ReID 요청)
            result2 = producers[camera_id]['track'].send_message(
                test_crop, 
                track_id=100 + i, 
                bbox=[100, 200, 50, 100],
                confidence=0.95,
                class_name="person"
            )
            print(f"  추적 결과 전송 결과: {result2}")
            
            # ReID 응답 (시뮬레이션)
            result3 = reid_response_producer.send_response(
                camera_id=camera_id,
                global_id=1000 + i,
                local_id=100 + i,
                elapsed=0.123,
                status='success'
            )
            print(f"  ReID 응답 전송 결과: {result3}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    finally:
        # 모든 Producer 종료
        for camera_id in camera_ids:
            producers[camera_id]['frame'].close()
            producers[camera_id]['track'].close()
        reid_response_producer.close()
        print("✅ 파티션 테스트 완료")