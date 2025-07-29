import cv2
import numpy as np
import json
import threading
import time
from kafka import KafkaConsumer
from dotenv import load_dotenv
import os
from collections import defaultdict, deque

class RealtimeVisualizer:
    """
    Kafka로부터 원본 프레임과 detection 결과를 받아서 실시간 시각화
    ReID 기능 제외, Detection + Tracking만 시각화
    """
    def __init__(self):
        # 환경 변수 로드
        env_file = 'env/aws.env'
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"[Visualizer] Loaded environment from: {env_file}")
        else:
            print(f"[Visualizer] Warning: Environment file not found: {env_file}")
        
        self.broker = os.getenv('BROKER', 'localhost:9092')
        self.frame_topic = os.getenv('FRAME_TOPIC', 'camera-frames')
        self.detection_topic = os.getenv('OUTPUT_TOPIC', 'output_topic')
        
        # 카메라별 최신 프레임과 detection 결과 저장
        self.latest_frames = {}
        self.latest_detections = {}
        self.frame_lock = threading.Lock()
        
        # 성능 모니터링
        self.fps_counters = defaultdict(lambda: deque(maxlen=30))
        self.processing_times = defaultdict(lambda: deque(maxlen=30))
        
        # 통계 정보
        self.stats = {
            'frames_received': 0,
            'detections_received': 0,
            'last_frame_time': {},
            'last_detection_time': {}
        }
        
        # 색상 팔레트 (트래킹 ID별) - 더 다양한 색상
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 128, 0),  # Olive
            (0, 128, 128),  # Teal
            (128, 0, 0),    # Maroon
            (255, 192, 203), # Pink
            (255, 69, 0),   # Red Orange
            (50, 205, 50),  # Lime Green
            (138, 43, 226)  # Blue Violet
        ]
        
        print(f"[Visualizer] Initialized - Broker: {self.broker}")
        print(f"[Visualizer] Topics - Frames: {self.frame_topic}, Detections: {self.detection_topic}")

    def get_color_for_id(self, track_id):
        """트래킹 ID에 따른 고유 색상 반환"""
        return self.colors[track_id % len(self.colors)]

    def draw_detection_info(self, frame, detections, camera_id):
        """프레임에 detection 정보 그리기"""
        if frame is None:
            return None
            
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 상단에 카메라 정보 표시 (더 큰 헤더)
        header_height = 80
        cv2.rectangle(display_frame, (0, 0), (w, header_height), (0, 0, 0), -1)
        
        # 카메라 ID
        cv2.putText(display_frame, f"Camera {camera_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # FPS 계산 및 표시
        current_time = time.time()
        self.fps_counters[camera_id].append(current_time)
        if len(self.fps_counters[camera_id]) > 1:
            fps = len(self.fps_counters[camera_id]) / (current_time - self.fps_counters[camera_id][0])
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (w-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 객체 수 표시
        object_count = len(detections) if detections else 0
        cv2.putText(display_frame, f"Objects: {object_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 타임스탬프 표시
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(current_time))
        cv2.putText(display_frame, timestamp_str, (w-150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Detection이 없으면 여기서 반환
        if not detections:
            return display_frame
        
        # 각 detection에 대해 bbox와 정보 그리기
        for detection in detections:
            try:
                track_id = detection.get('local_id', detection.get('track_id', 0))
                bbox = detection.get('bbox', [0, 0, 0, 0])
                score = detection.get('score', detection.get('confidence', 0.0))
                class_id = detection.get('class', detection.get('class_id', 0))
                is_new = detection.get('is_new', False)
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 경계 체크
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # 트래킹 ID에 따른 색상
                color = self.get_color_for_id(track_id)
                
                # 새로운 객체는 더 두꺼운 테두리
                thickness = 4 if is_new else 2
                
                # Bounding box 그리기
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨 생성
                label = f"ID:{track_id}"
                if score > 0:
                    label += f" ({score:.2f})"
                if is_new:
                    label += " [NEW]"
                    
                # 라벨 배경 크기 계산
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                label_height = 30
                
                # 라벨이 화면 밖으로 나가지 않도록 조정
                label_y = max(y1, label_height)
                
                # 라벨 배경
                cv2.rectangle(display_frame, 
                             (x1, label_y - label_height), 
                             (x1 + label_size[0] + 10, label_y), 
                             color, -1)
                
                # 라벨 텍스트
                cv2.putText(display_frame, label, (x1 + 5, label_y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 새로운 객체는 중앙에 점 표시
                if is_new:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(display_frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    cv2.circle(display_frame, (center_x, center_y), 12, (0, 255, 255), 2)
                
            except Exception as e:
                print(f"[Visualizer] Error drawing detection: {e}")
                continue
        
        return display_frame

    def consume_frames(self):
        """프레임 consumer 스레드"""
        try:
            consumer = KafkaConsumer(
                self.frame_topic,
                bootstrap_servers=[self.broker],
                key_deserializer=lambda b: b.decode('utf-8') if b else None,
                value_deserializer=lambda v: v,
                auto_offset_reset='latest',
                group_id='visualizer_frames',
                consumer_timeout_ms=1000  # 1초 타임아웃
            )
            
            print(f"[Visualizer] Frame consumer started for topic: {self.frame_topic}")
            
            for msg in consumer:
                camera_id = msg.key if msg.key else "unknown"
                
                try:
                    # 프레임 디코딩
                    frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.frame_lock:
                            self.latest_frames[camera_id] = frame
                            self.stats['frames_received'] += 1
                            self.stats['last_frame_time'][camera_id] = time.time()
                except Exception as e:
                    print(f"[Visualizer] Frame decode error for camera {camera_id}: {e}")
                    
        except Exception as e:
            print(f"[Visualizer] Frame consumer error: {e}")
            import traceback
            traceback.print_exc()

    def consume_detections(self):
        """Detection 결과 consumer 스레드"""
        try:
            consumer = KafkaConsumer(
                self.detection_topic,
                bootstrap_servers=[self.broker],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='visualizer_detections',
                consumer_timeout_ms=1000  # 1초 타임아웃
            )
            
            print(f"[Visualizer] Detection consumer started for topic: {self.detection_topic}")
            
            for msg in consumer:
                try:
                    data = msg.value
                    camera_id = data.get('camera_id', 'unknown')
                    tracks = data.get('tracks', [])
                    
                    with self.frame_lock:
                        self.latest_detections[camera_id] = {
                            'tracks': tracks,
                            'timestamp': data.get('timestamp', time.time())
                        }
                        self.stats['detections_received'] += 1
                        self.stats['last_detection_time'][camera_id] = time.time()
                        
                except Exception as e:
                    print(f"[Visualizer] Detection parse error: {e}")
                    
        except Exception as e:
            print(f"[Visualizer] Detection consumer error: {e}")
            import traceback
            traceback.print_exc()

    def create_status_display(self):
        """시스템 상태 표시 화면 생성"""
        status_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # 제목
        cv2.putText(status_frame, "Detection System Status", (200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 통계 정보
        y_pos = 120
        stats_text = [
            f"Frames Received: {self.stats['frames_received']}",
            f"Detections Received: {self.stats['detections_received']}",
            f"Active Cameras: {len(self.latest_frames)}",
            "",
            "Camera Status:"
        ]
        
        for text in stats_text:
            cv2.putText(status_frame, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40
        
        # 카메라별 상태
        current_time = time.time()
        for camera_id in sorted(self.latest_frames.keys()):
            last_frame = self.stats['last_frame_time'].get(camera_id, 0)
            last_detection = self.stats['last_detection_time'].get(camera_id, 0)
            
            frame_delay = current_time - last_frame if last_frame > 0 else 999
            detection_delay = current_time - last_detection if last_detection > 0 else 999
            
            status_color = (0, 255, 0) if frame_delay < 5 else (0, 255, 255) if frame_delay < 10 else (0, 0, 255)
            
            camera_text = f"  Camera {camera_id}: Frame {frame_delay:.1f}s ago, Detection {detection_delay:.1f}s ago"
            cv2.putText(status_frame, camera_text, (70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y_pos += 30
        
        # 사용법 안내
        y_pos += 40
        help_text = [
            "Controls:",
            "  ESC - Exit",
            "  S - Save Screenshot", 
            "  SPACE - Toggle Status View"
        ]
        
        for text in help_text:
            cv2.putText(status_frame, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_pos += 30
        
        return status_frame

    def run_visualization(self):
        """메인 시각화 루프"""
        print("[Visualizer] Starting visualization display")
        
        # Consumer 스레드들 시작
        frame_thread = threading.Thread(target=self.consume_frames, daemon=True)
        detection_thread = threading.Thread(target=self.consume_detections, daemon=True)
        
        frame_thread.start()
        detection_thread.start()
        
        print("[Visualizer] Consumer threads started")
        
        # 시각화 창 설정
        cv2.namedWindow('Multi-Camera Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Multi-Camera Detection', 1400, 900)
        
        no_data_count = 0
        show_status = False
        
        try:
            while True:
                current_time = time.time()
                
                with self.frame_lock:
                    camera_ids = set(self.latest_frames.keys()) | set(self.latest_detections.keys())
                
                if not camera_ids:
                    no_data_count += 1
                    if no_data_count % 50 == 0:  # 5초마다 로그
                        print("[Visualizer] Waiting for data...")
                    
                    # 대기 화면 표시
                    if show_status:
                        display_frame = self.create_status_display()
                    else:
                        display_frame = np.zeros((600, 800, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "Waiting for camera data...", (200, 250), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_frame, "Press SPACE to show status", (220, 350), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    
                    cv2.imshow('Multi-Camera Detection', display_frame)
                    
                else:
                    no_data_count = 0
                    
                    if show_status:
                        combined_frame = self.create_status_display()
                    else:
                        # 카메라별로 시각화된 프레임 생성
                        display_frames = []
                        
                        for camera_id in sorted(camera_ids):
                            with self.frame_lock:
                                frame = self.latest_frames.get(camera_id)
                                detection_data = self.latest_detections.get(camera_id, {})
                            
                            if frame is not None:
                                tracks = detection_data.get('tracks', [])
                                display_frame = self.draw_detection_info(frame, tracks, camera_id)
                                if display_frame is not None:
                                    display_frames.append(display_frame)
                            else:
                                # 프레임이 없으면 검은 화면에 카메라 ID만 표시
                                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(placeholder, f"Camera {camera_id}", (200, 220), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                cv2.putText(placeholder, "No Frame", (250, 260), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                display_frames.append(placeholder)
                        
                        if display_frames:
                            # 여러 카메라 화면을 하나로 합치기
                            if len(display_frames) == 1:
                                combined_frame = display_frames[0]
                            elif len(display_frames) == 2:
                                # 프레임 크기 맞추기
                                h1, w1 = display_frames[0].shape[:2]
                                h2, w2 = display_frames[1].shape[:2]
                                if h1 != h2 or w1 != w2:
                                    target_h, target_w = max(h1, h2), max(w1, w2)
                                    for i in range(len(display_frames)):
                                        display_frames[i] = cv2.resize(display_frames[i], (target_w, target_h))
                                combined_frame = np.hstack(display_frames)
                            else:
                                # 2x2 그리드로 배치
                                rows = []
                                for i in range(0, len(display_frames), 2):
                                    if i + 1 < len(display_frames):
                                        # 두 프레임 크기 맞추기
                                        h1, w1 = display_frames[i].shape[:2]
                                        h2, w2 = display_frames[i+1].shape[:2]
                                        target_h, target_w = max(h1, h2), max(w1, w2)
                                        frame1 = cv2.resize(display_frames[i], (target_w, target_h))
                                        frame2 = cv2.resize(display_frames[i+1], (target_w, target_h))
                                        row = np.hstack([frame1, frame2])
                                    else:
                                        # 홀수 개인 경우 빈 공간 추가
                                        blank = np.zeros_like(display_frames[i])
                                        row = np.hstack([display_frames[i], blank])
                                    rows.append(row)
                                
                                # 모든 행의 너비를 맞추기
                                if len(rows) > 1:
                                    max_width = max(row.shape[1] for row in rows)
                                    for i, row in enumerate(rows):
                                        if row.shape[1] < max_width:
                                            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                                            rows[i] = np.hstack([row, padding])
                                
                                combined_frame = np.vstack(rows)
                        else:
                            combined_frame = np.zeros((600, 800, 3), dtype=np.uint8)
                            cv2.putText(combined_frame, "No video data", (300, 300), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow('Multi-Camera Detection', combined_frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("[Visualizer] ESC pressed, exiting...")
                    break
                elif key == ord('s') or key == ord('S'):  # 스크린샷 저장
                    timestamp = int(time.time())
                    filename = f"detection_screenshot_{timestamp}.jpg"
                    if 'combined_frame' in locals():
                        cv2.imwrite(filename, combined_frame)
                        print(f"[Visualizer] Screenshot saved: {filename}")
                elif key == ord(' '):  # 스페이스바 - 상태 화면 토글
                    show_status = not show_status
                    status_text = "ON" if show_status else "OFF"
                    print(f"[Visualizer] Status display: {status_text}")
                
                # 50ms 대기 (20 FPS)
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[Visualizer] Stopping visualization...")
        except Exception as e:
            print(f"[Visualizer] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print("[Visualizer] Cleanup completed")

def main():
    """메인 함수"""
    print("[*] Starting Real-time Detection Visualizer (Detection Only)")
    
    try:
        visualizer = RealtimeVisualizer()
        visualizer.run_visualization()
    except Exception as e:
        print(f"[Error] Visualizer failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()