from kafka import KafkaConsumer
from kafka import TopicPartition
from dotenv import load_dotenv
import json
import cv2
import os
import base64
import numpy as np
from typing import Dict, Any, Optional, Callable, List
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseConsumer:
    """Kafka Consumer 기본 클래스 - 파티션 지원"""
    
    def __init__(self, topic_env_key: str, 
                 group_id: str,
                 auto_offset_reset: str = 'latest',
                 value_deserializer=None,
                 key_deserializer=None,
                 max_poll_records: int = 100,
                 enable_auto_commit: bool = True,
                 specific_partitions: Optional[List[int]] = None):
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
            specific_partitions: 특정 파티션만 구독 (None이면 모든 파티션)
        """
        # 환경변수 로드
        # 환경변수 로드
        dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
        load_dotenv(dotenv_path)
        
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv(topic_env_key)
        
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
        
        # 파티션 구독 설정
        if specific_partitions:
            # 특정 파티션만 구독
            partitions = [TopicPartition(self.topic, p) for p in specific_partitions]
            self.consumer.assign(partitions)
        else:
            # 모든 파티션 구독
            self.consumer.subscribe([self.topic])
        
        self._running = False
        self._thread = None
        self._message_count = 0
        self._specific_partitions = specific_partitions
        
        partition_info = f", Partitions: {specific_partitions}" if specific_partitions else ", All Partitions"
        logger.info(f"✅ {self.__class__.__name__} 초기화 완료 - Topic: {self.topic}, Group: {group_id}{partition_info}")
    
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
                            message_handler(message.key, message.value, topic_partition.partition)
                            
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
            "message_count": self._message_count,
            "specific_partitions": self._specific_partitions
        }
    
    def close(self):
        """Consumer 종료"""
        try:
            self.stop_consuming()
            self.consumer.close()
            logger.info(f"✅ {self.__class__.__name__} 종료 완료")
        except Exception as e:
            logger.error(f"❌ Consumer 종료 중 오류: {e}")


class CameraFrameConsumer(BaseConsumer):
    """특정 카메라의 프레임 수신용 Consumer - DetectorAndTracker와 연동"""
    
    def __init__(self, camera_id: str, group_id: Optional[str] = None):
        # 카메라별 고유한 그룹 ID 생성
        group_id = group_id or f"detector_group_{camera_id}"
        
        super().__init__(
            topic_env_key='FRAME_TOPIC',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: v  # 바이너리 데이터 그대로
        )
        self.camera_id = camera_id
    
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
    
    def start_frame_processing(self, detector_tracker, 
                              detection_result_handler: Optional[Callable[[str, Dict], None]] = None,
                              filter_camera_id: bool = True):
        """
        DetectorAndTracker를 사용한 프레임 처리 시작
        
        Args:
            detector_tracker: DetectorAndTracker 인스턴스
            detection_result_handler: 탐지 결과 처리 함수 (선택사항)
            filter_camera_id: 자신의 카메라 ID만 처리할지 여부
        """
        def message_handler(key: str, value: bytes, partition: int):
            # 카메라 ID 필터링
            if filter_camera_id and key != self.camera_id:
                return
            
            frame = self.decode_frame(value)
            if frame is not None:
                # DetectorAndTracker로 탐지 및 추적 수행
                boxes, scores, class_ids, timing_info = detector_tracker.infer(frame)
                
                # 추적 업데이트 (내부적으로 수행되며, 새로운 track 발견시 ReID 요청 전송)
                detector_tracker.detect_and_track(frame)
                
                # 결과를 핸들러로 전달 (선택사항)
                if detection_result_handler:
                    result_data = {
                        'camera_id': key,
                        'partition': partition,
                        'timestamp': int(time.time() * 1000),
                        'timing_info': timing_info,
                        'detection_count': len(boxes),
                        'detections': []
                    }
                    
                    # 탐지 결과 정리
                    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        x, y, w, h = box
                        class_name = detector_tracker.class_names[class_id] if class_id < len(detector_tracker.class_names) else f"Class_{class_id}"
                        
                        result_data['detections'].append({
                            'id': i,
                            'class_name': class_name,
                            'class_id': int(class_id),
                            'confidence': float(score),
                            'bbox': [float(x), float(y), float(w), float(h)]
                        })
                    
                    detection_result_handler(key, result_data)
            else:
                logger.warning(f"프레임 디코딩 실패 - Camera: {key}, Partition: {partition}")
        
        self.start_consuming(message_handler)


class GlobalReIDConsumer(BaseConsumer):
    """글로벌 ReID 요청 수신용 Consumer - 모든 카메라의 ReID 요청 처리"""
    
    def __init__(self, group_id: str = "global_reid_service"):
        super().__init__(
            topic_env_key='REID_REQUEST',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
        )
    
    def decode_crop_image(self, crop_data: Dict) -> Optional[np.ndarray]:
        """크롭 이미지 디코딩 - ReID 서비스 포맷 지원"""
        try:
            encoding = crop_data.get('encoding', 'base64')
            
            if encoding == 'base64':
                # ReID 서비스가 기대하는 'crop_jpg' 필드 확인
                crop_b64 = crop_data.get('crop_jpg', crop_data.get('crop_image', ''))
                if not crop_b64:
                    return None
                
                # Base64를 바이트로 디코딩
                crop_bytes = base64.b64decode(crop_b64)
                
                # 바이트를 numpy array로 변환 후 이미지 디코딩
                nparr = np.frombuffer(crop_bytes, np.uint8)
                crop_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                return crop_image
            
            elif encoding == 'binary':
                # 바이너리 방식은 별도 처리 필요
                logger.warning("바이너리 인코딩은 현재 지원되지 않습니다.")
                return None
            
            else:
                logger.error(f"지원하지 않는 인코딩 방식: {encoding}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 크롭 이미지 디코딩 실패: {e}")
            return None
    
    def start_reid_processing(self, reid_handler: Callable[[str, Dict, Optional[np.ndarray], int], None]):
        """ReID 요청 처리 시작 - 글로벌 ReID 서비스용"""
        def message_handler(key: str, value: Dict, partition: int):
            if value:
                crop_image = self.decode_crop_image(value)
                reid_handler(key, value, crop_image, partition)
            else:
                logger.warning(f"ReID 요청이 비어있음 - Camera: {key}, Partition: {partition}")
        
        self.start_consuming(message_handler)


class CameraReIDResponseConsumer(BaseConsumer):
    """특정 카메라의 ReID 응답 수신용 Consumer"""
    
    def __init__(self, camera_id: str, group_id: Optional[str] = None):
        # 카메라별 고유한 그룹 ID 생성
        group_id = group_id or f"reid_response_group_{camera_id}"
        
        super().__init__(
            topic_env_key='REID_RESPONSE',
            group_id=group_id,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
        )
        self.camera_id = camera_id
    
    def start_response_processing(self, response_handler: Callable[[str, Dict, int], None],
                                 filter_camera_id: bool = True):
        """ReID 응답 처리 시작"""
        def message_handler(key: str, value: Dict, partition: int):
            # 카메라 ID 필터링
            if filter_camera_id and key != self.camera_id:
                return
                
            if value:
                response_handler(key, value, partition)
            else:
                logger.warning(f"ReID 응답이 비어있음 - Camera: {key}, Partition: {partition}")
        
        self.start_consuming(message_handler)


# 파티션 기반 멀티 Consumer 매니저
class PartitionedConsumerManager:
    """파티션 기반 여러 Consumer를 통합 관리하는 클래스"""
    
    def __init__(self):
        self.consumers = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        self._running = False
    
    def add_camera_frame_consumer(self, camera_id: str, detector_tracker,
                                 detection_result_handler: Optional[Callable[[str, Dict], None]] = None,
                                 group_id: Optional[str] = None) -> CameraFrameConsumer:
        """
        카메라별 프레임 Consumer 추가 - DetectorAndTracker와 연동
        
        Args:
            camera_id: 카메라 ID
            detector_tracker: DetectorAndTracker 인스턴스
            detection_result_handler: 탐지 결과 처리 함수 (선택사항)
            group_id: 컨슈머 그룹 ID
        """
        consumer_id = f"frame_{camera_id}"
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        consumer = CameraFrameConsumer(camera_id, group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 결과 처리 (선택사항)
        async_handler = None
        if detection_result_handler:
            def async_detection_handler(camera_id: str, detection_data: Dict):
                self.executor.submit(detection_result_handler, camera_id, detection_data)
            async_handler = async_detection_handler
        
        consumer.start_frame_processing(detector_tracker, async_handler)
        return consumer
    
    def add_global_reid_consumer(self, reid_service_instance,
                                group_id: str = "global_reid_service") -> GlobalReIDConsumer:
        """
        글로벌 ReID Consumer 추가 - 모든 카메라의 ReID 요청 처리
        
        Args:
            reid_service_instance: ReID 서비스 인스턴스
            group_id: 컨슈머 그룹 ID
        """
        consumer_id = "global_reid"
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        consumer = GlobalReIDConsumer(group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 ReID 요청 처리
        def async_reid_handler(camera_id: str, reid_data: Dict, crop_image: Optional[np.ndarray], partition: int):
            self.executor.submit(self._process_reid_request, 
                               reid_service_instance, camera_id, reid_data, crop_image, partition)
        
        consumer.start_reid_processing(async_reid_handler)
        return consumer
    
    def add_camera_reid_response_consumer(self, camera_id: str, 
                                         response_handler: Callable[[str, Dict, int], None],
                                         group_id: Optional[str] = None) -> CameraReIDResponseConsumer:
        """
        카메라별 ReID 응답 Consumer 추가
        
        Args:
            camera_id: 카메라 ID
            response_handler: 응답 처리 함수
            group_id: 컨슈머 그룹 ID
        """
        consumer_id = f"reid_response_{camera_id}"
        if consumer_id in self.consumers:
            raise ValueError(f"Consumer ID '{consumer_id}'가 이미 존재합니다.")
        
        consumer = CameraReIDResponseConsumer(camera_id, group_id)
        self.consumers[consumer_id] = consumer
        
        # 비동기로 응답 처리
        def async_response_handler(camera_id: str, response_data: Dict, partition: int):
            self.executor.submit(response_handler, camera_id, response_data, partition)
        
        consumer.start_response_processing(async_response_handler)
        return consumer
    
    def _process_reid_request(self, reid_service, camera_id: str, reid_data: Dict, 
                            crop_image: Optional[np.ndarray], partition: int):
        """ReID 요청 처리 - ReID 서비스와 연동"""
        try:
            logger.info(f"🔍 ReID 요청 처리 시작 - Camera: {camera_id}, Partition: {partition}")
            
            # ReID 서비스의 process_reid_request 메서드 호출
            response = reid_service.process_reid_request(reid_data)
            
            # 응답을 ReID 응답 토픽으로 전송
            reid_service.send_response(response)
            
            logger.info(f"✅ ReID 처리 완료 - Camera: {camera_id}, Global ID: {response.get('global_id')}")
            
        except Exception as e:
            logger.error(f"❌ ReID 요청 처리 실패: {e}")
            # 에러 응답 전송
            error_response = {
                'camera_id': camera_id,
                'global_id': -1,
                'status': 'error',
                'error': str(e)
            }
            try:
                reid_service.send_response(error_response)
            except Exception as send_error:
                logger.error(f"❌ 에러 응답 전송 실패: {send_error}")
    
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
        logger.info("✅ PartitionedConsumerManager 종료 완료")


# 파티션 기반 사용 예시
def example_partitioned_usage():
    """파티션 기반 Consumer 사용 예시"""
    from detector_tracker import DetectorAndTracker  # 기존 코드
    from reid_service import ReIDService  # ReID 서비스
    
    # 여러 카메라 설정
    camera_ids = ["camera_001", "camera_002", "camera_003"]
    
    # DetectorAndTracker 인스턴스들 (카메라별)
    detectors = {}
    for camera_id in camera_ids:
        detectors[camera_id] = DetectorAndTracker(
            conf_threshold=0.25,
            iou_threshold=0.45,
            cameraID=camera_id
        )
    
    # 글로벌 ReID 서비스 (시스템에 1개)
    reid_service = ReIDService()
    
    # 탐지 결과 처리 함수
    def handle_detection_results(camera_id: str, detection_data: Dict):
        partition = detection_data.get('partition', -1)
        detection_count = detection_data.get('detection_count', 0)
        timing_info = detection_data.get('timing_info', {})
        
        logger.info(f"🎯 탐지 완료 - Camera: {camera_id}, Partition: {partition}, "
                   f"객체: {detection_count}개, 처리시간: {timing_info.get('total', 0):.3f}s")
    
    # ReID 응답 처리 함수
    def handle_reid_response(camera_id: str, response_data: Dict, partition: int):
        global_id = response_data.get('global_id', -1)
        local_id = response_data.get('local_id', -1)
        status = response_data.get('status', 'unknown')
        
        logger.info(f"🎯 ReID 응답 수신 - Camera: {camera_id}, Partition: {partition}, "
                   f"Local ID: {local_id} -> Global ID: {global_id}, Status: {status}")
    
    # Consumer Manager 생성
    manager = PartitionedConsumerManager()
    
    try:
        print("🚀 파티션 기반 시스템 시작...")
        
        # 각 카메라별 프레임 Consumer 추가
        for camera_id in camera_ids:
            manager.add_camera_frame_consumer(
                camera_id,
                detectors[camera_id],
                detection_result_handler=handle_detection_results,
                group_id=f"detector_group_{camera_id}"
            )
            
            # 각 카메라별 ReID 응답 Consumer 추가
            manager.add_camera_reid_response_consumer(
                camera_id,
                handle_reid_response,
                group_id=f"reid_response_group_{camera_id}"
            )
        
        # 글로벌 ReID Consumer 추가 (시스템에 1개)
        manager.add_global_reid_consumer(
            reid_service,
            group_id="global_reid_service"
        )
        
        print("✅ 모든 Consumer 시작 완료")
        print(f"📊 카메라 수: {len(camera_ids)}")
        print("🔄 파티션 기반 메시지 라우팅 활성화")
        
        # 실행
        time.sleep(60)  # 60초간 실행
        
        # 통계 정보 출력
        print("📊 최종 통계 정보:")
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("🛑 사용자 중단")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
    finally:
        print("🔚 모든 Consumer 종료 중...")
        manager.close_all()


if __name__ == "__main__":
    example_partitioned_usage()