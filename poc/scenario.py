#!/usr/bin/env python3
"""
DetectorAndTracker와 Kafka 완전 연동 예시

이 예시는 다음과 같은 파이프라인을 구현합니다:
1. 카메라 프레임 수신 (FrameConsumer)
2. YOLOv11 + DeepSort 처리 (DetectorAndTracker)
3. 검출/추적 결과 Kafka 전송 (Producer)
4. 결과 수신 및 처리 (Consumer)
"""

import cv2
import numpy as np
import time
import threading
import json
from typing import Dict, Optional, Any
import logging
import signal
import sys
from pathlib import Path

# 로컬 모듈 임포트
from tracking_module.new_dt import DetectorAndTracker
from kafka_util.improved_kafka_consumers import ConsumerManager, FrameConsumer
from kafka_util.improved_kafka_producers import dcreate_frame_producer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoStreamProcessor:
    """비디오 스트림 처리 및 Kafka 연동 클래스"""
    
    def __init__(self, camera_id: str, engine_path: str, 
                 video_source: Any = 0,  # 웹캠(0) 또는 비디오 파일 경로
                 enable_display: bool = True,
                 send_frames_to_kafka: bool = True):
        """
        초기화
        
        Args:
            camera_id: 카메라 ID
            engine_path: TensorRT 엔진 파일 경로
            video_source: 비디오 소스 (웹캠 번호 또는 파일 경로)
            enable_display: 화면 출력 여부
            send_frames_to_kafka: 프레임 Kafka 전송 여부
        """
        self.camera_id = camera_id
        self.video_source = video_source
        self.enable_display = enable_display
        self.send_frames_to_kafka = send_frames_to_kafka
        
        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {video_source}")
        
        # DetectorAndTracker 초기화
        self.detector = DetectorAndTracker(
            camera_id=camera_id,
            engine_path=engine_path,
            conf_threshold=0.25,
            iou_threshold=0.45,
            enable_kafka=True,
            enable_tracking=True,
            send_detection_results=True,
            send_tracking_results=True
        )
        
        # Kafka Producer (프레임 전송용)
        self.frame_producer = None
        if send_frames_to_kafka:
            try:
                self.frame_producer = create_frame_producer(camera_id)
                logger.info("✅ FrameProducer 초기화 완료")
            except Exception as e:
                logger.error(f"❌ FrameProducer 초기화 실패: {e}")
                self.send_frames_to_kafka = False
        
        # 통계 변수
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detection_count': 0,
            'tracking_count': 0,
            'avg_fps': 0.0,
            'avg_inference_time': 0.0
        }
        
        self._running = False
        self._start_time = time.time()
        
        logger.info(f"✅ VideoStreamProcessor 초기화 완료 - Camera: {camera_id}")
    
    def start_processing(self):
        """비디오 처리 시작"""
        if self._running:
            logger.warning("이미 처리 중입니다.")
            return
        
        self._running = True
        self._start_time = time.time()
        
        logger.info("🚀 비디오 처리 시작")
        
        try:
            while self._running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("프레임 읽기 실패")
                    break
                
                self.stats['total_frames'] += 1
                
                # 프레임 처리
                self._process_frame(frame)
                
                # FPS 계산
                elapsed_time = time.time() - self._start_time
                if elapsed_time > 0:
                    self.stats['avg_fps'] = self.stats['processed_frames'] / elapsed_time
                
                # ESC 키로 종료 (display 모드일 때)
                if self.enable_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                
        except Exception as e:
            logger.error(f"❌ 비디오 처리 중 오류: {e}")
        finally:
            self.stop_processing()
    
    def _process_frame(self, frame: np.ndarray):
        """단일 프레임 처리"""
        try:
            frame_start_time = time.time()
            
            # 1. DetectorAndTracker로 추론 및 추적
            tracks, timing_info = self.detector.detect_and_track(frame, debug=False)
            
            # 2. 통계 업데이트
            self.stats['processed_frames'] += 1
            if timing_info.get('total', 0) > 0:
                # 이동 평균으로 추론 시간 계산
                current_avg = self.stats['avg_inference_time']
                new_time = timing_info['total']
                alpha = 0.1  # 이동 평균 계수
                self.stats['avg_inference_time'] = current_avg * (1 - alpha) + new_time * alpha
            
            # 3. 프레임에 결과 그리기 (display용)
            if self.enable_display or self.send_frames_to_kafka:
                # 검출 결과가 있다면 그리기
                if hasattr(self.detector, '_last_boxes'):  # 마지막 검출 결과 사용
                    display_frame = self.detector.draw_detections(
                        frame, 
                        self.detector._last_boxes,
                        self.detector._last_scores, 
                        self.detector._last_class_ids
                    )
                else:
                    display_frame = frame.copy()
                
                # 추적 정보 그리기
                self._draw_tracking_info(display_frame, tracks, timing_info)
                
                # 화면 출력
                if self.enable_display:
                    cv2.imshow(f'Detection & Tracking - {self.camera_id}', display_frame)
                
                # Kafka로 결과 프레임 전송
                if self.send_frames_to_kafka and self.frame_producer:
                    try:
                        result = self.frame_producer.send_message(display_frame, quality=80)
                        if result.get('status_code') != 200:
                            logger.warning(f"프레임 전송 실패: {result.get('error')}")
                    except Exception as e:
                        logger.error(f"프레임 Kafka 전송 오류: {e}")
            
            # 4. 통계 업데이트
            if len(tracks) > 0:
                self.stats['tracking_count'] += len([t for t in tracks if t.is_confirmed()])
            
            frame_process_time = (time.time() - frame_start_time) * 1000
            
            # 주기적으로 통계 출력 (100프레임마다)
            if self.stats['processed_frames'] % 100 == 0:
                self._log_statistics()
                
        except Exception as e:
            logger.error(f"❌ 프레임 처리 오류: {e}")
    
    def _draw_tracking_info(self, frame: np.ndarray, tracks: list, timing_info: Dict):
        """추적 정보를 프레임에 그리기"""
        try:
            # 상단에 정보 표시
            info_y = 30
            cv2.putText(frame, f"Camera: {self.camera_id}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 25
            cv2.putText(frame, f"FPS: {self.stats['avg_fps']:.1f}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 25
            cv2.putText(frame, f"Inference: {timing_info.get('total', 0):.1f}ms", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 25
            active_tracks = len([t for t in tracks if t.is_confirmed()])
            cv2.putText(frame, f"Active Tracks: {active_tracks}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 각 트랙의 ID 표시
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                # 트랙 바운딩 박스
                l, t, r, b = track.to_ltrb()
                
                # 트랙 ID 표시
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (l, t-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        except Exception as e:
            logger.error(f"❌ 추적 정보 그리기 오류: {e}")
    
    def _log_statistics(self):
        """통계 정보 로깅"""
        detector_stats = self.detector.get_statistics()
        
        logger.info("📊 처리 통계:")
        logger.info(f"   총 프레임: {self.stats['total_frames']}")
        logger.info(f"   처리 프레임: {self.stats['processed_frames']}")
        logger.info(f"   평균 FPS: {self.stats['avg_fps']:.2f}")
        logger.info(f"   평균 추론 시간: {self.stats['avg_inference_time']:.2f}ms")
        logger.info(f"   활성 추적: {detector_stats['active_tracks']}")
    
    def stop_processing(self):
        """처리 중단"""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # 자원 정리
            if self.cap:
                self.cap.release()
            
            if self.enable_display:
                cv2.destroyAllWindows()
            
            if self.frame_producer:
                self.frame_producer.close()
            
            if self.detector:
                self.detector.cleanup()
            
            # 최종 통계 출력
            self._log_statistics()
            
            logger.info("✅ VideoStreamProcessor 종료 완료")
            
        except Exception as e:
            logger.error(f"❌ 종료 중 오류: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """전체 통계 정보 반환"""
        detector_stats = self.detector.get_statistics() if self.detector else {}
        
        return {
            "camera_id": self.camera_id,
            "video_stats": self.stats,
            "detector_stats": detector_stats,
            "running": self._running
        }


class KafkaResultMonitor:
    """Kafka 결과 모니터링 클래스"""
    
    def __init__(self):
        self.consumer_manager = ConsumerManager()
        self.detection_count = 0
        self.tracking_count = 0
        
    def start_monitoring(self):
        """모니터링 시작"""
        logger.info("🔍 Kafka 결과 모니터링 시작")
        
        # 검출 결과 모니터링
        self.consumer_manager.add_detection_consumer(
            "monitor_detection",
            self._handle_detection_result,
            group_id="monitor_detection_group"
        )
        
        # 추적 결과 모니터링
        self.consumer_manager.add_tracking_consumer(
            "monitor_tracking",
            self._handle_tracking_result,
            group_id="monitor_tracking_group"
        )
    
    def _handle_detection_result(self, camera_id: str, detection_data: Dict):
        """검출 결과 처리"""
        self.detection_count += 1
        detections = detection_data.get('detections', [])
        
        if self.detection_count % 50 == 0:  # 50개마다 로그
            logger.info(f"📊 검출 결과 - Camera: {camera_id}, "
                       f"객체 수: {len(detections)}, "
                       f"총 검출 메시지: {self.detection_count}")
    
    def _handle_tracking_result(self, camera_id: str, tracking_data: Dict, 
                               crop_image: Optional[np.ndarray]):
        """추적 결과 처리"""
        self.tracking_count += 1
        track_id = tracking_data.get('track_id')
        class_name = tracking_data.get('class_name', 'unknown')
        
        if self.tracking_count % 10 == 0:  # 10개마다 로그
            logger.info(f"🎯 추적 결과 - Camera: {camera_id}, "
                       f"Track ID: {track_id}, "
                       f"클래스: {class_name}, "
                       f"총 추적 메시지: {self.tracking_count}")
            
            if crop_image is not None:
                logger.info(f"   크롭 이미지 크기: {crop_image.shape}")
    
    def stop_monitoring(self):
        """모니터링 중단"""
        self.consumer_manager.close_all()
        logger.info("🛑 Kafka 결과 모니터링 중단")
    
    def get_statistics(self) -> Dict[str, Any]:
        """모니터링 통계"""
        return {
            "detection_messages": self.detection_count,
            "tracking_messages": self.tracking_count,
            "consumer_stats": self.consumer_manager.get_statistics()
        }


def main():
    """메인 실행 함수"""
    # 설정
    CAMERA_ID = "camera_001"
    ENGINE_PATH = "yolo_engine/yolo11m_fp16.engine"  # 실제 경로로 수정 필요
    VIDEO_SOURCE = "./sample.mp4"# 웹캠 사용, 파일의 경우 경로 입력 
    
    # 엔진 파일 존재 확인
    if not Path(ENGINE_PATH).exists():
        logger.error(f"❌ TensorRT 엔진 파일을 찾을 수 없습니다: {ENGINE_PATH}")
        logger.info("💡 ENGINE_PATH를 실제 엔진 파일 경로로 수정해주세요.")
        return
    
    processor = None
    monitor = None
    
    def signal_handler(sig, frame):
        """시그널 핸들러"""
        logger.info("🛑 종료 신호 수신")
        if processor:
            processor.stop_processing()
        if monitor:
            monitor.stop_monitoring()
        sys.exit(0)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Kafka 결과 모니터링 시작 (선택사항)
        monitor = KafkaResultMonitor()
        monitor.start_monitoring()
        
        # 비디오 처리 시작
        processor = VideoStreamProcessor(
            camera_id=CAMERA_ID,
            engine_path=ENGINE_PATH,
            video_source=VIDEO_SOURCE,
            enable_display=True,
            send_frames_to_kafka=True
        )
        
        # 별도 스레드에서 통계 출력
        def print_stats():
            while True:
                time.sleep(30)  # 30초마다
                if processor:
                    stats = processor.get_statistics()
                    logger.info("📊 전체 통계:")
                    logger.info(f"   비디오: {json.dumps(stats['video_stats'], indent=2)}")
                
                if monitor:
                    monitor_stats = monitor.get_statistics()
                    logger.info(f"   모니터링: {json.dumps(monitor_stats, indent=2)}")
        
        stats_thread = threading.Thread(target=print_stats, daemon=True)
        stats_thread.start()
        
        # 메인 처리 루프
        processor.start_processing()
        
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
    finally:
        logger.info("🔚 프로그램 종료")
        if processor:
            processor.stop_processing()
        if monitor:
            monitor.stop_monitoring()


if __name__ == "__main__":
    main()