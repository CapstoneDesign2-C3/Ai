import os
import cv2
import json
import time
import threading
import numpy as np
import traceback
from kafka_util.producers import FrameProducer, DetectedResultProducer
from kafka_util.consumers import FrameConsumer, DetectedResultConsumer

class KafkaTestSuite:
    """
    Producer와 Consumer를 테스트하는 종합 테스트 클래스 (향상된 에러 처리 포함)
    """
    
    def __init__(self):
        self.test_camera_id = 1
        self.test_results = {
            'frame_tests': [],
            'detection_tests': []
        }
        
        # 연결 상태 확인
        self.frame_producer = None
        self.detection_producer = None
        self.frame_consumer = None
        self.detection_consumer = None
    
    def initialize_components(self):
        """Kafka 컴포넌트 초기화 및 연결 상태 확인"""
        print("🔌 Initializing Kafka components...")
        
        try:
            # Producer 초기화
            print("   - Initializing Frame Producer...")
            self.frame_producer = FrameProducer(self.test_camera_id)
            print("   ✅ Frame Producer initialized")
            
            print("   - Initializing Detection Producer...")
            self.detection_producer = DetectedResultProducer()
            print("   ✅ Detection Producer initialized")
            
            # Consumer 초기화
            print("   - Initializing Frame Consumer...")
            self.frame_consumer = FrameConsumer()
            if hasattr(self.frame_consumer, 'consumer') and self.frame_consumer.consumer is not None:
                print("   ✅ Frame Consumer initialized")
            else:
                print("   ❌ Frame Consumer initialization failed - consumer is None")
                return False
            
            print("   - Initializing Detection Consumer...")
            self.detection_consumer = DetectedResultConsumer()
            if hasattr(self.detection_consumer, 'consumer') and self.detection_consumer.consumer is not None:
                print("   ✅ Detection Consumer initialized")
            else:
                print("   ❌ Detection Consumer initialization failed - consumer is None")
                return False
                
            return True
            
        except Exception as e:
            print(f"   ❌ Component initialization failed: {e}")
            print(f"   📍 Error details: {traceback.format_exc()}")
            return False
    
    def check_kafka_connection(self):
        """Kafka 브로커 연결 상태 확인"""
        print("🔍 Checking Kafka connection...")
        
        try:
            # Producer 연결 테스트
            if self.frame_producer:
                # 더미 메시지로 연결 테스트
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                result = self.frame_producer.send_message(test_frame)
                print(f"   Frame Producer test: {result}")
            
            if self.detection_producer:
                test_data = json.dumps({"test": "connection"}).encode('utf-8')
                result = self.detection_producer.send_message("test", test_data)
                print(f"   Detection Producer test: {result}")
            
            # Consumer 연결 테스트 (토픽 존재 확인)
            if self.frame_consumer and hasattr(self.frame_consumer.consumer, 'list_topics'):
                topics = self.frame_consumer.consumer.list_topics(timeout=5)
                print(f"   Available topics: {list(topics.topics.keys())}")
                
            return True
            
        except Exception as e:
            print(f"   ❌ Connection check failed: {e}")
            return False
    
    def create_test_frame(self, width=640, height=480, text="Test Frame"):
        """테스트용 가짜 프레임 생성"""
        # 컬러 배경 생성
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 100), font, 0.7, (255, 255, 255), 2)
        
        # 컬러 박스들 추가 (detection bbox 시뮬레이션)
        cv2.rectangle(frame, (100, 150), (200, 300), (0, 255, 0), 2)  # 초록색 박스
        cv2.rectangle(frame, (250, 200), (350, 350), (255, 0, 0), 2)  # 파란색 박스
        
        return frame
    
    def create_test_detections(self):
        """테스트용 detection 결과 생성"""
        detections = [
            {
                'bbox': [100, 150, 100, 150],  # [x, y, w, h]
                'confidence': 0.85,
                'class': 0,  # person
                'track_id': 1
            },
            {
                'bbox': [250, 200, 100, 150],
                'confidence': 0.92,
                'class': 0,
                'track_id': 2
            }
        ]
        return detections
    
    def test_frame_producer_consumer(self, num_frames=5):
        """프레임 송수신 테스트 (향상된 에러 처리)"""
        print("=== Frame Producer-Consumer Test ===")
        
        if not self.frame_producer or not self.frame_consumer:
            print("❌ Frame components not properly initialized")
            return False
        
        received_frames = []
        consumer_error = None
        
        def consume_frames():
            """프레임 수신 스레드"""
            nonlocal consumer_error
            frame_count = 0
            
            try:
                # Consumer가 None인지 확인
                if self.frame_consumer.consumer is None:
                    consumer_error = "Consumer is None"
                    return
                
                print("   📥 Consumer thread started, waiting for messages...")
                
                # 타임아웃 설정
                timeout_start = time.time()
                timeout_duration = 30  # 30초 타임아웃
                
                for message in self.frame_consumer.consumer:
                    # 타임아웃 체크
                    if time.time() - timeout_start > timeout_duration:
                        print("   ⏰ Consumer timeout reached")
                        break
                    
                    camera_id = message.key
                    frame_bytes = message.value
                    
                    print(f"   📨 Received message from {camera_id}: {len(frame_bytes)} bytes")
                    
                    # JPEG 바이트를 OpenCV 이미지로 디코딩
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        received_frames.append({
                            'camera_id': camera_id,
                            'frame_shape': frame.shape,
                            'timestamp': time.time()
                        })
                        
                        print(f"   ✅ Received frame from {camera_id}: {frame.shape}")
                        
                        # 프레임을 파일로 저장 (선택적)
                        if not os.path.exists('test_output'):
                            os.makedirs('test_output')
                        cv2.imwrite(f'test_output/received_frame_{frame_count}.jpg', frame)
                        
                        frame_count += 1
                        if frame_count >= num_frames:
                            print(f"   🎯 Received all {num_frames} frames, stopping consumer")
                            break
                    else:
                        print("   ❌ Failed to decode frame")
                        
            except Exception as e:
                consumer_error = f"Consumer error: {e}"
                print(f"   ❌ {consumer_error}")
                print(f"   📍 Error details: {traceback.format_exc()}")
        
        # Consumer 스레드 시작
        consumer_thread = threading.Thread(target=consume_frames, daemon=True)
        consumer_thread.start()
        
        # 잠시 대기 후 프레임 전송 시작
        print("   ⏱️  Waiting 3 seconds before sending frames...")
        time.sleep(3)
        
        # 테스트 프레임 전송
        print(f"   📤 Sending {num_frames} test frames...")
        send_success_count = 0
        
        for i in range(num_frames):
            try:
                test_frame = self.create_test_frame(text=f"Frame {i+1}")
                
                result = self.frame_producer.send_message(test_frame)
                if result and result.get('status_code') == 200:
                    print(f"   ✅ Sent frame {i+1}")
                    send_success_count += 1
                else:
                    print(f"   ❌ Failed to send frame {i+1}: {result}")
                
                time.sleep(1)  # 1초 간격으로 전송
                
            except Exception as e:
                print(f"   ❌ Error sending frame {i+1}: {e}")
        
        # Consumer 스레드 완료 대기
        print("   ⏱️  Waiting for consumer to finish...")
        consumer_thread.join(timeout=15)
        
        # 결과 정리
        self.test_results['frame_tests'] = received_frames
        print(f"\n   📊 Frame Test Results:")
        print(f"      - Sent: {send_success_count}/{num_frames} frames")
        print(f"      - Received: {len(received_frames)} frames")
        if send_success_count > 0:
            print(f"      - Success Rate: {len(received_frames)/send_success_count*100:.1f}%")
        
        if consumer_error:
            print(f"      - Consumer Error: {consumer_error}")
        
        return len(received_frames) > 0 and consumer_error is None
    
    def test_detection_producer_consumer(self, num_detections=3):
        """Detection 결과 송수신 테스트 (향상된 에러 처리)"""
        print("\n=== Detection Producer-Consumer Test ===")
        
        if not self.detection_producer or not self.detection_consumer:
            print("❌ Detection components not properly initialized")
            return False
        
        received_detections = []
        consumer_error = None
        
        def consume_detections():
            """Detection 수신 스레드"""
            nonlocal consumer_error
            detection_count = 0
            
            try:
                # Consumer가 None인지 확인
                if self.detection_consumer.consumer is None:
                    consumer_error = "Consumer is None"
                    return
                
                print("   📥 Detection consumer thread started, waiting for messages...")
                
                # 타임아웃 설정
                timeout_start = time.time()
                timeout_duration = 30  # 30초 타임아웃
                
                for message in self.detection_consumer.consumer:
                    # 타임아웃 체크
                    if time.time() - timeout_start > timeout_duration:
                        print("   ⏰ Detection consumer timeout reached")
                        break
                    
                    camera_id = message.key
                    detection_bytes = message.value
                    
                    print(f"   📨 Received detection message from {camera_id}")
                    
                    # JSON 파싱
                    try:
                        detection_data = json.loads(detection_bytes.decode('utf-8'))
                        
                        received_detections.append({
                            'camera_id': camera_id,
                            'detections': detection_data,
                            'timestamp': time.time()
                        })
                        
                        detections_count = len(detection_data.get('detections', []))
                        print(f"   ✅ Received detection from {camera_id}: {detections_count} objects")
                        
                        detection_count += 1
                        if detection_count >= num_detections:
                            print(f"   🎯 Received all {num_detections} detections, stopping consumer")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"   ❌ Failed to parse detection JSON: {e}")
                        
            except Exception as e:
                consumer_error = f"Detection consumer error: {e}"
                print(f"   ❌ {consumer_error}")
                print(f"   📍 Error details: {traceback.format_exc()}")
        
        # Consumer 스레드 시작
        consumer_thread = threading.Thread(target=consume_detections, daemon=True)
        consumer_thread.start()
        
        # 잠시 대기 후 detection 전송 시작
        print("   ⏱️  Waiting 3 seconds before sending detections...")
        time.sleep(3)
        
        # 테스트 detection 전송
        print(f"   📤 Sending {num_detections} test detections...")
        send_success_count = 0
        
        for i in range(num_detections):
            try:
                test_detections = self.create_test_detections()
                
                # Detection 데이터를 JSON으로 패키징
                payload = {
                    'timestamp': time.time(),
                    'frame_id': f"frame_{i+1}",
                    'detections': test_detections
                }
                
                payload_json = json.dumps(payload).encode('utf-8')
                
                result = self.detection_producer.send_message(self.test_camera_id, payload_json)
                if result and result.get('status_code') == 200:
                    print(f"   ✅ Sent detection {i+1}")
                    send_success_count += 1
                else:
                    print(f"   ❌ Failed to send detection {i+1}: {result}")
                
                time.sleep(1)  # 1초 간격으로 전송
                
            except Exception as e:
                print(f"   ❌ Error sending detection {i+1}: {e}")
        
        # Consumer 스레드 완료 대기
        print("   ⏱️  Waiting for detection consumer to finish...")
        consumer_thread.join(timeout=15)
        
        # 결과 정리
        self.test_results['detection_tests'] = received_detections
        print(f"\n   📊 Detection Test Results:")
        print(f"      - Sent: {send_success_count}/{num_detections} detections")
        print(f"      - Received: {len(received_detections)} detections")
        if send_success_count > 0:
            print(f"      - Success Rate: {len(received_detections)/send_success_count*100:.1f}%")
        
        if consumer_error:
            print(f"      - Consumer Error: {consumer_error}")
        
        return len(received_detections) > 0 and consumer_error is None
    
    def test_concurrent_operations(self):
        """동시 송수신 테스트"""
        print("\n=== Concurrent Operations Test ===")
        
        if not self.frame_producer:
            print("❌ Frame producer not available for concurrent test")
            return False
        
        # 여러 카메라 시뮬레이션
        camera_ids = ["cam_01", "cam_02", "cam_03"]
        
        def send_frames_for_camera(camera_id, num_frames=3):
            try:
                producer = FrameProducer(camera_id)
                success_count = 0
                
                for i in range(num_frames):
                    frame = self.create_test_frame(text=f"{camera_id} Frame {i+1}")
                    result = producer.send_message(frame)
                    
                    if result and result.get('status_code') == 200:
                        print(f"   📤 {camera_id}: Sent frame {i+1}")
                        success_count += 1
                    else:
                        print(f"   ❌ {camera_id}: Failed to send frame {i+1}")
                    
                    time.sleep(0.5)
                
                print(f"   📊 {camera_id}: {success_count}/{num_frames} frames sent successfully")
                
            except Exception as e:
                print(f"   ❌ {camera_id}: Error in concurrent sending: {e}")
        
        # 동시에 여러 카메라에서 프레임 전송
        threads = []
        for camera_id in camera_ids:
            thread = threading.Thread(target=send_frames_for_camera, args=(camera_id,))
            threads.append(thread)
            thread.start()
        
        # 모든 전송 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        print("   ✅ Concurrent operations test completed")
        return True
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 Starting Enhanced Kafka Producer-Consumer Test Suite")
        print("=" * 60)
        
        # 출력 디렉토리 생성
        os.makedirs('test_output', exist_ok=True)
        
        try:
            # 1. 컴포넌트 초기화
            if not self.initialize_components():
                print("❌ Failed to initialize Kafka components. Check your Kafka setup.")
                return False
            
            # 2. 연결 상태 확인
            if not self.check_kafka_connection():
                print("⚠️  Kafka connection issues detected. Proceeding with limited testing.")
            
            # 3. 개별 테스트 실행
            print("\n" + "=" * 60)
            frame_success = self.test_frame_producer_consumer(num_frames=3)
            detection_success = self.test_detection_producer_consumer(num_detections=3)
            
            # 4. 동시 작업 테스트
            concurrent_success = self.test_concurrent_operations()
            
            # 5. 전체 결과 요약
            print("\n" + "=" * 60)
            print("📋 TEST SUMMARY")
            print("=" * 60)
            print(f"Component Init:  {'✅ PASS' if (self.frame_producer and self.detection_producer) else '❌ FAIL'}")
            print(f"Frame Test:      {'✅ PASS' if frame_success else '❌ FAIL'}")
            print(f"Detection Test:  {'✅ PASS' if detection_success else '❌ FAIL'}")
            print(f"Concurrent Test: {'✅ PASS' if concurrent_success else '❌ FAIL'}")
            
            overall_success = frame_success and detection_success and concurrent_success
            print(f"Overall Result:  {'🎉 ALL TESTS PASSED' if overall_success else '⚠️  SOME TESTS FAILED'}")
            
            # 디버깅 정보 출력
            if not overall_success:
                print("\n🔍 DEBUGGING INFORMATION:")
                print("   - Check if Kafka broker is running")
                print("   - Verify topic creation and permissions")
                print("   - Check network connectivity")
                print("   - Review Kafka consumer group settings")
                print("   - Examine log files for detailed error messages")
            
            return overall_success
            
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
            print(f"📍 Error details: {traceback.format_exc()}")
            return False

# 개별 진단 함수들
def diagnose_kafka_setup():
    """Kafka 설정 진단"""
    print("🔍 Kafka Setup Diagnosis")
    print("=" * 40)
    
    try:
        # 1. Producer 테스트
        print("1. Testing Frame Producer...")
        producer = FrameProducer("diagnostic_test")
        if producer:
            print("   ✅ Frame Producer created successfully")
        else:
            print("   ❌ Frame Producer creation failed")
        
        # 2. Consumer 테스트
        print("2. Testing Frame Consumer...")
        consumer = FrameConsumer()
        if consumer and hasattr(consumer, 'consumer') and consumer.consumer:
            print("   ✅ Frame Consumer created successfully")
            
            # 토픽 리스트 가져오기 시도
            try:
                topics = consumer.consumer.list_topics(timeout=10)
                print(f"   📋 Available topics: {list(topics.topics.keys())}")
            except Exception as e:
                print(f"   ⚠️  Could not list topics: {e}")
        else:
            print("   ❌ Frame Consumer creation failed or consumer is None")
            
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        print(f"📍 Details: {traceback.format_exc()}")

def test_simple_frame_send():
    """간단한 프레임 전송 테스트 (향상된 버전)"""
    print("📤 Enhanced Simple Frame Send Test")
    print("=" * 40)
    
    try:
        producer = FrameProducer("test_camera")
        if not producer:
            print("❌ Failed to create producer")
            return
        
        # 테스트 프레임 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Hello Kafka!", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        print("Sending test frame...")
        result = producer.send_message(frame)
        
        print(f"📊 Result: {result}")
        
        if result and result.get('status_code') == 200:
            print("✅ Frame sent successfully!")
        else:
            print("❌ Frame sending failed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"📍 Details: {traceback.format_exc()}")

def test_simple_detection_send():
    """간단한 detection 전송 테스트 (향상된 버전)"""
    print("📤 Enhanced Simple Detection Send Test")
    print("=" * 40)
    
    try:
        producer = DetectedResultProducer()
        if not producer:
            print("❌ Failed to create detection producer")
            return
        
        test_data = {
            'timestamp': time.time(),
            'detections': [
                {'bbox': [100, 100, 50, 100], 'confidence': 0.9, 'class': 0}
            ]
        }
        
        payload = json.dumps(test_data).encode('utf-8')
        
        print("Sending test detection...")
        result = producer.send_message("test_camera", payload)
        
        print(f"📊 Result: {result}")
        
        if result and result.get('status_code') == 200:
            print("✅ Detection sent successfully!")
        else:
            print("❌ Detection sending failed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"📍 Details: {traceback.format_exc()}")

def test_frame_consumer_only():
    """프레임 수신만 테스트 (향상된 버전)"""
    print("📥 Enhanced Frame Consumer Test")
    print("=" * 40)
    print("Waiting for messages... (Press Ctrl+C to stop)")
    
    try:
        consumer = FrameConsumer()
        
        if not consumer or not hasattr(consumer, 'consumer') or consumer.consumer is None:
            print("❌ Consumer initialization failed or consumer is None")
            return
        
        message_count = 0
        start_time = time.time()
        
        for message in consumer.consumer:
            camera_id = message.key
            frame_bytes = message.value
            
            message_count += 1
            elapsed = time.time() - start_time
            
            print(f"📨 [{message_count}] Received from {camera_id}: {len(frame_bytes)} bytes (t={elapsed:.1f}s)")
            
            # 이미지 디코딩 테스트
            try:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    print(f"   ✅ Successfully decoded frame: {frame.shape}")
                    filename = f"received_{camera_id}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"   💾 Saved as: {filename}")
                else:
                    print("   ❌ Failed to decode frame")
                    
            except Exception as decode_error:
                print(f"   ❌ Decode error: {decode_error}")
            
            # 5개 메시지 받으면 자동 종료
            if message_count >= 5:
                print("🎯 Received 5 messages, stopping test")
                break
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Test stopped by user. Received {message_count} messages.")
    except Exception as e:
        print(f"❌ Consumer test failed: {e}")
        print(f"📍 Details: {traceback.format_exc()}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "diagnose":
            diagnose_kafka_setup()
        elif test_type == "send_frame":
            test_simple_frame_send()
        elif test_type == "send_detection":
            test_simple_detection_send()
        elif test_type == "consume_frame":
            test_frame_consumer_only()
        elif test_type == "full":
            test_suite = KafkaTestSuite()
            success = test_suite.run_all_tests()
            sys.exit(0 if success else 1)
        else:
            print("Usage: python test_kafka_enhanced.py [diagnose|send_frame|send_detection|consume_frame|full]")
    else:
        # 기본값: 진단부터 시작
        print("Starting with Kafka diagnosis...")
        diagnose_kafka_setup()
        
        print("\nRunning full test suite...")
        test_suite = KafkaTestSuite()
        success = test_suite.run_all_tests()
        
        sys.exit(0 if success else 1)