import cv2
import numpy as np
import torch
import gc
import time
import sys
import os
import json
import threading
from contextlib import contextmanager
from kafka import KafkaProducer, KafkaConsumer
from dotenv import load_dotenv

# CUDA 관련 imports
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    print("⚠️  PyCUDA not available, using PyTorch CUDA only")
    PYCUDA_AVAILABLE = False

# 프로젝트 경로 추가
sys.path.append('/home/hiperwall/Ai_modules/new/poc')

from poc.tracking_module.tracker_v2 import OptimizedDetectorTracker
from poc.nvr_util.nvr_client import NVRClient
from poc.nvr_util.exceptions import NVRConnectionError, NVRRecieveError


@contextmanager
def cuda_context_manager():
    """CUDA 컨텍스트 자동 관리"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        try:
            # CUDA 캐시 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # 컨텍스트 워밍업
            warmup_tensor = torch.zeros(1).cuda()
            del warmup_tensor
            torch.cuda.synchronize()
            
            yield device
            
        finally:
            # 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    else:
        yield None


class FrameProducer:
    """Kafka를 통한 프레임 전송 클래스"""
    
    def __init__(self, broker: str, topic: str = 'camera-frames'):
        self.broker = broker
        self.topic = topic
        self.producer = None
        self._init_producer()
    
    def _init_producer(self):
        """Kafka Producer 초기화"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.broker],
                value_serializer=lambda v: v,  # 바이너리 데이터는 그대로 전송
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                batch_size=16384 * 4,  # 프레임 데이터가 크므로 배치 크기 증가
                linger_ms=5,
                compression_type='lz4'  # 압축 추가
            )
            print(f"✅ Kafka Producer initialized - Broker: {self.broker}, Topic: {self.topic}")
        except Exception as e:
            print(f"❌ Failed to initialize Kafka Producer: {e}")
            raise
    
    def send_frame(self, camera_id: str, frame: np.ndarray) -> bool:
        """프레임을 Kafka로 전송"""
        try:
            # 프레임을 JPEG로 인코딩
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]  # 품질 85%
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            # Kafka로 전송
            future = self.producer.send(
                self.topic,
                key=camera_id,
                value=frame_bytes
            )
            
            # 비동기 결과 확인 (선택적)
            # future.get(timeout=1)  # 동기 모드로 전환하려면 주석 해제
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send frame to Kafka: {e}")
            return False
    
    def close(self):
        """Producer 정리"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print("📡 Kafka Producer closed")


class ResultConsumer:
    """처리 결과를 받아서 시각화하는 클래스"""
    
    def __init__(self, broker: str, topic: str = 'detection-results'):
        self.broker = broker
        self.topic = topic
        self.consumer = None
        self.is_running = False
        self.latest_results = {}
        self._init_consumer()
    
    def _init_consumer(self):
        """Kafka Consumer 초기화"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=[self.broker],
                key_deserializer=lambda b: b.decode('utf-8') if b else None,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',  # 최신 메시지부터
                enable_auto_commit=True,
                group_id='visualization-client',
                consumer_timeout_ms=100  # 짧은 타임아웃으로 반응성 향상
            )
            print(f"✅ Kafka Consumer initialized - Topic: {self.topic}")
        except Exception as e:
            print(f"❌ Failed to initialize Kafka Consumer: {e}")
            raise
    
    def start_consuming(self):
        """백그라운드에서 결과 수신 시작"""
        self.is_running = True
        self.consume_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.consume_thread.start()
        print("🔄 Started consuming detection results in background")
    
    def _consume_loop(self):
        """결과 수신 루프"""
        while self.is_running:
            try:
                for message in self.consumer:
                    if not self.is_running:
                        break
                    
                    camera_id = message.key or "unknown"
                    result_data = message.value
                    
                    # 최신 결과 저장
                    self.latest_results[camera_id] = result_data
                    
            except Exception as e:
                if self.is_running:
                    print(f"⚠️  Consumer error: {e}")
                    time.sleep(1)
    
    def get_latest_results(self, camera_id: str):
        """특정 카메라의 최신 결과 반환"""
        return self.latest_results.get(camera_id, None)
    
    def stop(self):
        """Consumer 중지"""
        self.is_running = False
        if hasattr(self, 'consume_thread'):
            self.consume_thread.join(timeout=2)
        if self.consumer:
            self.consumer.close()
            print("📡 Kafka Consumer closed")


class VisualizationEngine:
    """객체 탐지 결과 시각화 엔진"""
    
    @staticmethod
    def draw_detection_results(frame: np.ndarray, results: dict) -> np.ndarray:
        """탐지 결과를 프레임에 시각화"""
        if not results or 'tracks' not in results:
            return frame
        
        annotated_frame = frame.copy()
        tracks = results['tracks']
        
        # 정보 표시 영역
        info_y = 30
        cv2.putText(annotated_frame, f"Camera: {results.get('camera_id', 'Unknown')}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Objects: {len(tracks)}", 
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각 객체 시각화
        for track in tracks:
            try:
                bbox = track.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 바운딩 박스 색상 결정
                is_new = track.get('is_new', False)
                color = (0, 255, 255) if is_new else (0, 255, 0)  # 새 객체는 노란색, 기존은 초록색
                thickness = 3 if is_new else 2
                
                # 바운딩 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨 정보 구성
                label_parts = ["Person"]
                
                local_id = track.get('local_id')
                if local_id is not None:
                    label_parts.append(f"ID:{local_id}")
                
                score = track.get('score')
                if score is not None:
                    label_parts.append(f"{score:.2f}")
                
                if is_new:
                    label_parts.append("NEW")
                
                label = " ".join(label_parts)
                
                # 라벨 배경 그리기
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_bg_color = (0, 255, 255) if is_new else (0, 255, 0)
                
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), 
                             label_bg_color, -1)
                
                # 라벨 텍스트 그리기
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            except Exception as viz_error:
                print(f"⚠️  Visualization error: {viz_error}")
                continue
        
        # 타임스탬프 표시
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_str, 
                   (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame


def load_environment():
    """환경 변수 로드"""
    try:
        load_dotenv('env/aws.env')
        broker = os.getenv('BROKER')
        if not broker:
            raise ValueError("BROKER environment variable not set")
        
        frame_topic = os.getenv('FRAME_TOPIC', 'camera-frames')
        result_topic = os.getenv('OUTPUT_TOPIC', 'detection-results')
        
        print(f"✅ Environment loaded - Broker: {broker}")
        return broker, frame_topic, result_topic
        
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        raise


def select_camera_channel(nvr_client):
    """사용할 카메라 채널 선택"""
    if not nvr_client.NVRChannelList:
        raise Exception("No camera channels available")
    
    if len(nvr_client.NVRChannelList) == 1:
        return nvr_client.NVRChannelList[0]
    
    print("📹 Available cameras:")
    for i, channel in enumerate(nvr_client.NVRChannelList):
        print(f"  {i}: Camera {channel.camera_id} ({channel.camera_ip})")
    
    # 기본적으로 첫 번째 카메라 선택
    selected_index = 0
    return nvr_client.NVRChannelList[selected_index]


def main():
    print("🚀 Starting Refactored NVR YOLO Visualization with Kafka...")
    
    # 환경 변수 로드
    broker, frame_topic, result_topic = load_environment()
    
    # Kafka 컴포넌트 초기화
    frame_producer = None
    result_consumer = None
    nvr_client = None
    channel = None
    
    try:
        with cuda_context_manager() as cuda_device:
            if cuda_device is not None:
                print(f"✅ CUDA context initialized on device {cuda_device}")
            else:
                print("⚠️  Running on CPU mode")
            
            # Kafka 컴포넌트 초기화
            frame_producer = FrameProducer(broker, frame_topic)
            result_consumer = ResultConsumer(broker, result_topic)
            
            # 결과 수신 시작
            result_consumer.start_consuming()
            
            # NVR 클라이언트 초기화
            print("📡 Connecting to NVR...")
            nvr_client = NVRClient()
            
            # 카메라 채널 선택 및 연결
            channel = select_camera_channel(nvr_client)
            camera_id = str(channel.camera_id)
            print(f"📹 Selected camera: {camera_id} ({channel.camera_ip})")
            
            try:
                channel.connect()
                print(f"✅ Connected to camera: {camera_id}")
                print(f"📊 Camera info - Resolution: {channel.width}x{channel.height}, FPS: {channel.fps:.1f}")
            except NVRConnectionError as e:
                raise Exception(f"NVR connection failed: {e}")
            
            # 시각화 엔진 초기화
            viz_engine = VisualizationEngine()
            
            print("🎥 Starting video processing... Press 'q' to quit")
            
            frame_count = 0
            start_time = time.time()
            connection_retry_count = 0
            max_retries = 5
            kafka_send_interval = 2  # 2프레임마다 Kafka로 전송
            
            while True:
                try:
                    # 프레임 가져오기
                    ret, frame = channel.cap.read()
                    if not ret or frame is None:
                        print("⚠️  No frame received, attempting reconnection...")
                        
                        if connection_retry_count < max_retries:
                            try:
                                channel.disconnect()
                                time.sleep(1)
                                channel.connect()
                                connection_retry_count += 1
                                print(f"🔄 Reconnection attempt {connection_retry_count}/{max_retries}")
                                continue
                            except Exception as reconnect_error:
                                print(f"❌ Reconnection failed: {reconnect_error}")
                                connection_retry_count += 1
                                time.sleep(2)
                                if connection_retry_count >= max_retries:
                                    raise Exception("Max reconnection attempts reached")
                                continue
                        else:
                            raise Exception("Max reconnection attempts reached")
                    
                    connection_retry_count = 0
                    frame_count += 1
                    
                    # Kafka로 프레임 전송 (간격 조절)
                    if frame_count % kafka_send_interval == 0:
                        success = frame_producer.send_frame(camera_id, frame)
                        if not success:
                            print("⚠️  Failed to send frame to Kafka")
                    
                    # 처리 결과 가져오기
                    detection_results = result_consumer.get_latest_results(camera_id)
                    
                    # 시각화
                    if detection_results:
                        annotated_frame = viz_engine.draw_detection_results(frame, detection_results)
                        
                        # 주기적으로 탐지 정보 출력
                        if frame_count % 100 == 0:
                            tracks = detection_results.get('tracks', [])
                            new_objects = sum(1 for t in tracks if t.get('is_new', False))
                            print(f"🎯 Frame {frame_count}: {len(tracks)} objects ({new_objects} new)")
                    else:
                        annotated_frame = frame
                    
                    # FPS 정보 표시
                    if frame_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        print(f"📊 Frame: {frame_count}, FPS: {fps:.1f}")
                    
                    # 화면에 표시
                    cv2.imshow('NVR YOLO Detection with Kafka', annotated_frame)
                    
                    # 키 입력 확인
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("🛑 Quit signal received")
                        break
                        
                except NVRRecieveError as nvr_error:
                    print(f"⚠️  NVR receive error: {nvr_error}")
                    time.sleep(0.1)
                    continue
                except Exception as frame_error:
                    print(f"⚠️  Frame processing error: {frame_error}")
                    time.sleep(0.1)
                    continue
                    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        
    finally:
        print("🧹 Cleaning up resources...")
        
        # 리소스 정리
        if result_consumer:
            result_consumer.stop()
        
        if frame_producer:
            frame_producer.close()
        
        if channel:
            try:
                channel.disconnect()
                print("📡 NVR channel disconnected")
            except Exception as e:
                print(f"⚠️  NVR disconnect warning: {e}")
        
        # OpenCV 윈도우 정리
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")


if __name__ == "__main__":
    main()