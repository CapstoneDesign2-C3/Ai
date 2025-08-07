from kafka import KafkaConsumer
from dotenv import load_dotenv
import json
import cv2
import os
import base64
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseConsumer:
    """Kafka Consumer 기본 클래스"""
    
    def __init__(self, topic_env_key: str, 
                 group_id: str,
                 auto_offset_reset: str = 'latest',
                 value_deserializer=None,
                 key_deserializer=None,
                 max_poll_records: int = 100,
                 enable_auto_commit: bool = True):
        """
        기본 Consumer 초기화
        
        Args:
            topic_env_key: 환경변수에서 읽을 토픽 키
            group_id: 컨슈머 그룹 ID
            auto_offset_reset: 오프셋 리셋 정책
            value_deserializer: 값 역직렬화 함수
            key_deserializer: 키 역직렬화 함수
            max_poll_records: 한 번에 가져올 최대 레코드 수
            enable_auto_commit: 자동 커밋 여부
        """
        # 환경변수 로드
        dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
        load_dotenv(dotenv_path)
        
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('FRAME_TOPIC')
        
        if not self.broker or not self.topic:
            raise ValueError(f"BROKER 또는 {topic_env_key} 환경변수가 설정되지 않았습니다.")
        
        # Consumer 설정
        consumer_config = {
            'bootstrap_servers': self.broker,
            'group_id': group_id,
            'auto_offset_reset': auto_offset_reset,
            'api_version': (2, 5, 0),
            'max_poll_records': max_poll_records,
            'enable_auto_commit': enable_auto_commit,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000,
        }
        
        if key_deserializer:
            consumer_config['key_deserializer'] = key_deserializer
        if value_deserializer:
            consumer_config['value_deserializer'] = value_deserializer
            
        self.consumer = KafkaConsumer(**consumer_config)
        self.consumer.subscribe([self.topic])
        
        self._running = False
        self._thread = None
        self._message_count = 0
        
        logger.info(f"✅ {self.__class__.__name__} 초기화 완료 - Topic: {self.topic}, Group: {group_id}")
    
    def start_consuming(self, message_handler: Callable[[Any, Any], None], 
                       error_handler: Optional[Callable[[Exception], None]] = None):
        """메시지 소비 시작"""
        if self._running:
            logger.warning("Consumer가 이미 실행 중입니다.")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop, 
            args=(message_handler, error_handler)
        )
        self._thread.daemon = True
        self._thread.start()
        
        logger.info(f"🚀 {self.__class__.__name__} 시작됨")
    
    def stop_consuming(self):
        """메시지 소비 중단"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info(f"🛑 {self.__class__.__name__} 중단됨")
    
    def _consume_loop(self, message_handler: Callable, error_handler: Optional[Callable]):
        """메시지 소비 루프"""
        while self._running:
            try:
                # 메시지 폴링 (타임아웃 1초)
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        if not self._running:
                            break
                        
                        try:
                            self._message_count += 1
                            message_handler(message.key, message.value)
                            
                        except Exception as e:
                            logger.error(f"❌ 메시지 처리 오류: {e}")
                            if error_handler:
                                error_handler(e)
                
            except Exception as e:
                logger.error(f"❌ Consumer 오류: {e}")
                if error_handler:
                    error_handler(e)
                time.sleep(1)  # 오류 시 잠깐 대기
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "topic": self.topic,
            "is_running": self._running,
            "message_count": self._message_count
        }
    
    def close(self):
        """Consumer 종료"""
        try:
            self.stop_consuming()
            self.consumer.close()
            logger.info(f"✅ {self.__class__.__name__} 종료 완료")
        except Exception as e:
            logger.error(f"❌ Consumer 종료 중 오류: {e}")


class FrameConsumer(BaseConsumer):
    """프레임 수신용 Consumer"""
    
    def __init__(self, group_id: str = "frame_consumer_group"):
        super().__init__(
            topic_env_key='camera-frames',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: v  # 바이너리 데이터 그대로
        )
    
    def decode_frame(self, frame_bytes: bytes) -> Optional[np.ndarray]:
        """프레임 바이트를 OpenCV 이미지로 디코딩"""
        try:
            # bytes를 numpy array로 변환
            nparr = np.frombuffer(frame_bytes, np.uint8)
            # JPEG 디코딩
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"❌ 프레임 디코딩 실패: {e}")
            return None
    
    def start_frame_processing(self, frame_handler: Callable[[str, np.ndarray], None]):
        """프레임 처리 시작"""
        def message_handler(key: str, value: bytes):
            frame = self.decode_frame(value)
            if frame is not None:
                frame_handler(key, frame)
            else:
                logger.warning(f"프레임 디코딩 실패 - Camera: {key}")
        
        self.start_consuming(message_handler)


class DetectedResultConsumer(BaseConsumer):
    """검출 결과 수신용 Consumer"""
    
    def __init__(self, group_id: str = "detection_result_consumer_group"):
        super().__init__(
            topic_env_key='detected_result',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
        )
    
    def start_detection_processing(self, detection_handler: Callable[[str, Dict], None]):
        """검출 결과 처리 시작"""
        def message_handler(key: str, value: Dict):
            if value:
                detection_handler(key, value)
            else:
                logger.warning(f"검출 결과가 비어있음 - Camera: {key}")
        
        self.start_consuming(message_handler)


class TrackResultConsumer(BaseConsumer):
    """추적 결과 수신용 Consumer"""
    
    def __init__(self, group_id: str = "track_result_consumer_group"):
        super().__init__(
            topic_env_key='REID_REQUEST',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
        )
    
    def decode_crop_image(self, crop_data: Dict) -> Optional[np.ndarray]:
        """크롭 이미지 디코딩"""
        try:
            encoding = crop_data.get('encoding', 'base64')
            
            if encoding == 'base64':
                # Base64 디코딩
                crop_b64 = crop_data.get('crop_image', '')
                if not crop_b64:
                    return None
                
                # Base64를 바이트로 디코딩
                crop_bytes = base64.b64decode(crop_b64)
                
                # 바이트를 numpy array로 변환 후 이미지 디코딩
                nparr = np.frombuffer(crop_bytes, np.uint8)
                crop_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                return crop_image
            
            elif encoding == 'binary':
                # 바이너리 방식은 별도 처리 필요 (구현에 따라 다름)
                logger.warning("바이너리 인코딩은 현재 지원되지 않습니다.")
                return None
            
            else:
                logger.error(f"지원하지 않는 인코딩 방식: {encoding}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 크롭 이미지 디코딩 실패: {e}")
            return None
    
    def start_tracking_processing(self, tracking_handler: Callable[[str, Dict, Optional[np.ndarray]], None]):
        """추적 결과 처리 시작"""
        def message_handler(key: str, value: Dict):
            if value:
                crop_image = self.decode_crop_image(value)
                tracking_handler(key, value, crop_image)
            else:
                logger.warning(f"추적 결과가 비어있음 - Camera: {key}")
        
        self.start_consuming(message_handler)


# 멀티 Consumer 매니저
class ConsumerManager:
    """여러 Consumer를 통합 관리하는 클래스"""
    
    def __init__(self):
        self.consumers = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
    
    def add_frame_consumer(self, consumer_id: str, frame_handler: Callable[[str, np.ndarray], None],
                          group_id: Optional[str] = None) -> FrameConsumer:
        """프레임 Consumer 추가"""
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        group_id = group_id or f"frame_consumer_{consumer_id}"
        consumer = FrameConsumer(group_id=group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 프레임 처리
        def async_frame_handler(camera_id: str, frame: np.ndarray):
            self.executor.submit(frame_handler, camera_id, frame)
        
        consumer.start_frame_processing(async_frame_handler)
        return consumer
    
    def add_detection_consumer(self, consumer_id: str, detection_handler: Callable[[str, Dict], None],
                              group_id: Optional[str] = None) -> DetectedResultConsumer:
        """검출 결과 Consumer 추가"""
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        group_id = group_id or f"detection_consumer_{consumer_id}"
        consumer = DetectedResultConsumer(group_id=group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 검출 결과 처리
        def async_detection_handler(camera_id: str, detection_data: Dict):
            self.executor.submit(detection_handler, camera_id, detection_data)
        
        consumer.start_detection_processing(async_detection_handler)
        return consumer
    
    def add_tracking_consumer(self, consumer_id: str, 
                             tracking_handler: Callable[[str, Dict, Optional[np.ndarray]], None],
                             group_id: Optional[str] = None) -> TrackResultConsumer:
        """추적 결과 Consumer 추가"""
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        group_id = group_id or f"tracking_consumer_{consumer_id}"
        consumer = TrackResultConsumer(group_id=group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 추적 결과 처리
        def async_tracking_handler(camera_id: str, tracking_data: Dict, crop_image: Optional[np.ndarray]):
            self.executor.submit(tracking_handler, camera_id, tracking_data, crop_image)
        
        consumer.start_tracking_processing(async_tracking_handler)
        return consumer
    
    def get_consumer(self, consumer_id: str) -> Optional[BaseConsumer]:
        """Consumer 가져오기"""
        return self.consumers.get(consumer_id)
    
    def remove_consumer(self, consumer_id: str) -> bool:
        """Consumer 제거"""
        if consumer_id in self.consumers:
            consumer = self.consumers[consumer_id]
            consumer.close()
            del self.consumers[consumer_id]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계 정보"""
        stats = {
            "total_consumers": len(self.consumers),
            "executor_info": {
                "max_workers": self.executor._max_workers,
                "active_threads": len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
            },
            "consumers": {}
        }
        
        for consumer_id, consumer in self.consumers.items():
            stats["consumers"][consumer_id] = consumer.get_statistics()
        
        return stats
    
    def close_all(self):
        """모든 Consumer 종료"""
        for consumer_id, consumer in self.consumers.items():
            try:
                consumer.close()
                logger.info(f"✅ Consumer '{consumer_id}' 종료 완료")
            except Exception as e:
                logger.error(f"❌ Consumer '{consumer_id}' 종료 중 오류: {e}")
        
        self.consumers.clear()
        self.executor.shutdown(wait=True)
        logger.info("✅ ConsumerManager 종료 완료")


# 사용 예시
def example_usage():
    """Consumer 사용 예시"""
    
    # 프레임 처리 함수
    def handle_frame(camera_id: str, frame: np.ndarray):
        print(f"📽️ 프레임 수신 - Camera: {camera_id}, Shape: {frame.shape}")
        # 여기서 DetectorAndTracker로 처리할 수 있음
        # detector.detect_and_track(frame)
    
    # 검출 결과 처리 함수
    def handle_detection(camera_id: str, detection_data: Dict):
        print(f"🎯 검출 결과 수신 - Camera: {camera_id}")
        print(f"   검출 개수: {detection_data.get('detection_count', 0)}")
        print(f"   타이밍: {detection_data.get('timing_info', {})}")
        
        # 검출된 객체들 정보 출력
        for i, detection in enumerate(detection_data.get('detections', [])):
            print(f"   객체 {i}: {detection.get('class_name')} ({detection.get('confidence', 0):.2f})")
    
    # 추적 결과 처리 함수
    def handle_tracking(camera_id: str, tracking_data: Dict, crop_image: Optional[np.ndarray]):
        print(f"🎯 추적 결과 수신 - Camera: {camera_id}")
        print(f"   Track ID: {tracking_data.get('track_id')}")
        print(f"   클래스: {tracking_data.get('class_name')}")
        print(f"   신뢰도: {tracking_data.get('confidence')}")
        
        if crop_image is not None:
            print(f"   크롭 이미지: {crop_image.shape}")
            # ReID 시스템으로 전송하거나 저장 가능
        else:
            print("   크롭 이미지 디코딩 실패")
    
    # Consumer Manager 생성
    manager = ConsumerManager()
    
    try:
        # 각종 Consumer 추가
        print("🚀 Consumer들 시작...")
        
        frame_consumer = manager.add_frame_consumer(
            "main_frame_consumer", 
            handle_frame, 
            group_id="frame_processor_group"
        )
        
        detection_consumer = manager.add_detection_consumer(
            "main_detection_consumer", 
            handle_detection,
            group_id="detection_processor_group"
        )
        
        tracking_consumer = manager.add_tracking_consumer(
            "main_tracking_consumer", 
            handle_tracking,
            group_id="tracking_processor_group"
        )
        
        print("✅ 모든 Consumer 시작 완료")
        
        # 실행 (실제 환경에서는 메인 루프나 서비스로 실행)
        time.sleep(10)  # 10초간 실행
        
        # 통계 정보 출력
        print("📊 통계 정보:")
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2))
        
    except KeyboardInterrupt:
        print("🛑 사용자 중단")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
    finally:
        print("🔚 Consumer들 종료 중...")
        manager.close_all()


if __name__ == "__main__":
    example_usage()