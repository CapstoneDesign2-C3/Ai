import os
import cv2
import time
from dotenv import load_dotenv
from poc.tracking_module.detection_and_tracking import DetectorAndTracker
import os
import time
import threading
import cv2
import numpy as np
from dotenv import load_dotenv


from nvr_util.nvr_client import NVRChannel, NVRClient  # requires nvr_util, db_util packages to be importable

# 우리가 앞서 만든 프레임 컨슈머 (프레임 ← Kafka 수신 담당)
from kafka_util.consumers import create_frame_consumer

def main():
    # 1) ENV 세팅: TensorRT 엔진(.plan) 경로를 지정
    #    예: export ENGINE_PATH=/path/to/yolov11.engine
    dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
    load_dotenv(dotenv_path)
    engine_path = os.getenv('ENGINE_PATH')
    if engine_path is None:
        print("❌ ENGINE_PATH 환경변수가 설정되지 않았습니다.")
        return

    # 2) DetectorAndracker 초기화
    #    class_names_path: 클래스 이름 JSON 또는 TXT 파일 경로 (없으면 COCO 기본)
    detector = DetectorAndTracker(
        class_names_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45,
        cameraID="TEST_CAM01"
    )

    # 3) (선택) ReID embedder 주입
    #    실제 ReID 모델이 없다면, 더미 임베딩으로 대체
    #    detector.embedder = DummyEmbedder()  # 실제 구현체로 교체
    detector.embedder = lambda crops: [ [0.0]*128 for _ in crops ]
    client = NVRClient()
    channel = client.NVRChannelList[0]
    channel.connect()
    
    # 4) 비디오 열기
    cap = cv2.VideoCapture('sample.mp4')
    if not cap.isOpened():
        print("❌ sample.mp4 를 열 수 없습니다.")
        return

    # 5) 프레임별 처리
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # a) 검출·추적 실행 (detect_and_track 내부에서 ReID 메시지 송신)
        detector.detect_and_track(frame, debug=False)

        # b) 시각화: infer → draw_detections 순으로 별도 처리
        boxes, scores, class_ids, timings = detector.infer(frame)
        vis = detector.draw_detections(frame, boxes, scores, class_ids)

        # c) 처리 시간 출력
        total_ms = (timings['total'] * 1000)
        cv2.putText(vis, f"Time: {total_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # d) 결과 디스플레이
        cv2.imshow('Detection & Tracking', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
