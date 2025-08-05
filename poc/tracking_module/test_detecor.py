#!/usr/bin/env python3
"""
Detector 테스트 유틸리티
개발 및 디버깅을 위한 추가 도구들
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime
import cv2
import numpy as np
from kafka import KafkaConsumer
from dotenv import load_dotenv


class DetectionResultMonitor:
    """검출 결과 모니터링"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='detection_results'):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='latest'
        )
        self.stats = {}
        
    def monitor(self, duration=60):
        """결과 모니터링"""
        print(f"📊 검출 결과 모니터링 시작 ({duration}초)")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                message = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message.items():
                    for msg in messages:
                        self.process_detection_result(msg.key, msg.value)
                        
        except KeyboardInterrupt:
            print("\n🛑 모니터링 중단")
        finally:
            self.print_summary()
            
    def process_detection_result(self, camera_id, result):
        """검출 결과 처리"""
        if camera_id not in self.stats:
            self.stats[camera_id] = {
                'total_frames': 0,
                'total_detections': 0,
                'classes': {},
                'avg_inference_time': 0,
                'last_update': None
            }
            
        stats = self.stats[camera_id]
        stats['total_frames'] += 1
        stats['total_detections'] += len(result.get('detections', []))
        stats['last_update'] = datetime.now()
        
        # 추론 시간 업데이트
        if 'inference_time_ms' in result:
            current_avg = stats['avg_inference_time']
            new_time = result['inference_time_ms']
            stats['avg_inference_time'] = (current_avg * (stats['total_frames'] - 1) + new_time) / stats['total_frames']
            
        # 클래스별 통계
        for detection in result.get('detections', []):
            class_name = detection.get('class_name', 'unknown')
            if class_name not in stats['classes']:
                stats['classes'][class_name] = 0
            stats['classes'][class_name] += 1
            
        # 실시간 출력
        if stats['total_frames'] % 10 == 0:
            print(f"📷 {camera_id}: {stats['total_frames']}프레임, "
                  f"{stats['total_detections']}검출, "
                  f"평균 {stats['avg_inference_time']:.1f}ms")
            
    def print_summary(self):
        """통계 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 검출 결과 요약")
        print(f"{'='*60}")
        
        for camera_id, stats in self.stats.items():
            print(f"\n📷 카메라: {camera_id}")
            print(f"   총 프레임: {stats['total_frames']}")
            print(f"   총 검출: {stats['total_detections']}")
            print(f"   평균 추론 시간: {stats['avg_inference_time']:.1f}ms")
            print(f"   마지막 업데이트: {stats['last_update']}")
            
            if stats['classes']:
                print("   검출 클래스:")
                for class_name, count in sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {class_name}: {count}개")


class SimulatedCameraFeeder:
    """시뮬레이션 카메라 피더 (테스트용)"""
    
    def __init__(self, video_path=None, camera_count=4):
        self.video_path = video_path
        self.camera_count = camera_count
        
        # Kafka Producer 설정
        from poc.kafka_util.producers import FrameProducer
        self.frame_producer = FrameProducer()
        
    def feed_from_video(self, fps=5):
        """비디오 파일에서 프레임 공급"""
        if not self.video_path or not os.path.exists(self.video_path):
            print("❌ 비디오 파일을 찾을 수 없습니다")
            return
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ 비디오 파일을 열 수 없습니다")
            return
            
        frame_interval = 1.0 / fps
        frame_count = 0
        
        print(f"🎬 비디오 파일에서 {self.camera_count}개 가상 카메라로 전송 시작")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 비디오 끝나면 처음부터 다시
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                # 여러 카메라로 시뮬레이션
                for i in range(self.camera_count):
                    camera_id = f"sim_camera_{i+1}"
                    
                    # 약간의 변형 추가 (노이즈, 밝기 등)
                    modified_frame = self.add_variation(frame, i)
                    
                    # JPEG 인코딩
                    _, buffer = cv2.imencode('.jpg', modified_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    # Kafka 전송
                    self.frame_producer.send_message(camera_id, frame_bytes)
                    
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"📤 {frame_count}프레임 전송됨 ({self.camera_count}개 카메라)")
                    
                time.sleep(frame_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 시뮬레이션 중단")
        finally:
            cap.release()
            
    def add_variation(self, frame, camera_index):
        """카메라별 변형 추가"""
        modified = frame.copy()
        
        # 카메라별로 다른 변형
        if camera_index == 0:
            # 밝기 조정
            modified = cv2.convertScaleAbs(modified, alpha=1.1, beta=10)
        elif camera_index == 1:
            # 가우시안 블러
            modified = cv2.GaussianBlur(modified, (3, 3), 0)
        elif camera_index == 2:
            # 색상 조정
            hsv = cv2.cvtColor(modified, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = hsv[:,:,1] * 1.2  # 채도 증가
            modified = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # camera_index == 3은 원본 유지
        
        return modified
        
    def feed_from_webcam(self, fps=5):
        """웹캠에서 프레임 공급"""
        cap = cv2.VideoCapture(0)  # 기본 웹캠
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다")
            return
            
        frame_interval = 1.0 / fps
        frame_count = 0
        
        print(f"📷 웹캠에서 {self.camera_count}개 가상 카메라로 전송 시작")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 웹캠 프레임 읽기 실패")
                    time.sleep(1)
                    continue
                    
                # 여러 카메라로 시뮬레이션
                for i in range(self.camera_count):
                    camera_id = f"webcam_sim_{i+1}"
                    
                    # 카메라별 ROI 생성 (화면 분할)
                    h, w = frame.shape[:2]
                    if self.camera_count == 4:
                        # 2x2 그리드로 분할
                        row = i // 2
                        col = i % 2
                        roi = frame[row*h//2:(row+1)*h//2, col*w//2:(col+1)*w//2]
                    else:
                        roi = frame
                        
                    # JPEG 인코딩
                    _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    # Kafka 전송
                    self.frame_producer.send_message(camera_id, frame_bytes)
                    
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"📤 {frame_count}프레임 전송됨 ({self.camera_count}개 카메라)")
                    
                time.sleep(frame_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 시뮬레이션 중단")
        finally:
            cap.release()


def main():
    """유틸리티 메인 함수"""
    parser = argparse.ArgumentParser(description='Detector 테스트 유틸리티')
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 결과 모니터링
    monitor_parser = subparsers.add_parser('monitor', help='검출 결과 모니터링')
    monitor_parser.add_argument('--duration', type=int, default=60, help='모니터링 시간(초)')
    monitor_parser.add_argument('--topic', default='detection_results', help='Kafka 토픽')
    
    # 시뮬레이션 카메라
    sim_parser = subparsers.add_parser('simulate', help='시뮬레이션 카메라 실행')
    sim_parser.add_argument('--source', choices=['video', 'webcam'], required=True, help='소스 타입')
    sim_parser.add_argument('--video', help='비디오 파일 경로 (source=video일 때)')
    sim_parser.add_argument('--cameras', type=int, default=4, help='가상 카메라 수')
    sim_parser.add_argument('--fps', type=int, default=5, help='전송 FPS')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # 환경변수 로드
    load_dotenv('env/aws.env')
    
    if args.command == 'monitor':
        monitor = DetectionResultMonitor(topic=args.topic)
        monitor.monitor(duration=args.duration)
        
    elif args.command == 'simulate':
        feeder = SimulatedCameraFeeder(camera_count=args.cameras)
        
        if args.source == 'video':
            if not args.video:
                print("❌ --video 옵션이 필요합니다")
                return 1
            feeder.video_path = args.video
            feeder.feed_from_video(fps=args.fps)
        elif args.source == 'webcam':
            feeder.feed_from_webcam(fps=args.fps)
            
    return 0


if __name__ == "__main__":
    sys.exit(main())