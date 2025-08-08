import cv2
from nvr_util.exceptions import NVRConnectionError, NVRRecieveError
import uuid
from db_util.db_util import PostgreSQL
from kafka_util import producers
from dotenv import load_dotenv
import os
import time
import logging

logger = logging.getLogger(__name__)

class NVRChannel:
    def __init__(self, camera_id, camera_ip, camera_port, stream_path, rtsp_id, rtsp_password):
        self.camera_id = camera_id
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.stream_path = stream_path
        self.rtsp_id = rtsp_id
        self.rtsp_password = rtsp_password
        self.isRecording = False
        self.frame_producer = producers.create_frame_producer(self.camera_id)

    def connect(self):
        rtsp_live_url = f'rtsp://{self.rtsp_id}:{self.rtsp_password}@{self.camera_ip}:{self.camera_port}{self.stream_path}'
        self.cap = cv2.VideoCapture(rtsp_live_url)
        if not self.cap.isOpened():
            raise NVRConnectionError()

        # 파일 저장을 위한 변수 저장
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("successfully connected.\n")

    def disconnect(self):
        self.cap.release()
    
    def receive(self):
        # 5 FPS
        send_count = 0
        send_interval = 1.0 / 5.0
        next_send_at = time.monotonic()  # monotonic clock

        consecutive_fail = 0
        MAX_FAIL = 50  # 연속 실패 허용치 (상황에 맞게 조정)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= MAX_FAIL:
                    raise NVRRecieveError(f"프레임 연속 {MAX_FAIL}회 읽기 실패")
                time.sleep(0.02)  # 잠깐 쉬고 재시도
                continue
            consecutive_fail = 0

            # 선택: 녹화 중이면 파일에도 저장
            if getattr(self, "isRecording", False):
                try:
                    self.out.write(frame)
                except Exception as e:
                    logger.warning(f"Recording write 실패: {e}")

            try:
                # Kafka로 JPEG bytes 전송 (key=camera_id, value=bytes)
                # 주의: FrameProducer.send_message 내부에서 flush를 매번 하지 않는 것을 권장
                res = self.frame_producer.send_message(frame=frame)
                send_count += 1
                if send_count % 10 == 0:  # 10프레임마다 한 번
                    logger.info(f"[NVR] sent {send_count} frames from {self.camera_id}")
                if isinstance(res, dict) and res.get('status_code') != 200:
                    logger.error(f"Kafka send 실패: {res}")
            except Exception as e:
                logger.error(f"❌ 프레임 전송 실패: {e}")


    def startRecord(self):
        self.file_name = f"{uuid.uuid4()}.mp4"
        self.out = cv2.VideoWriter(self.file_name, self.fourcc, self.fps, (self.width, self.height))
        self.isRecording = True
    
    def endRecord(self):
        self.isRecording = False
        self.out.release()

        return self.file_name
    
class NVRClient:
    def __init__(self):
        dotenv_path = '/home/hiperwall/Ai_modules/Ai/env/aws.env'
        load_dotenv(dotenv_path)
        db = PostgreSQL(os.getenv('DB_HOST'), os.getenv('DB_NAME'), os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_PORT'))
        camera_list = db.getCameraInfo()
        
        self.NVRChannelList = []
        for camera in camera_list:
            self.NVRChannelList.append(NVRChannel(**camera))

