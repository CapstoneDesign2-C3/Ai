import cv2
from poc.nvr_util.exceptions import NVRConnectionError, NVRRecieveError
import uuid
from poc.db_util.db_util import PostgreSQL
from dotenv import load_dotenv
import os

class NVRChannel:
    def __init__(self, camera_id, camera_ip, camera_port, stream_path, rtsp_id, rtsp_password):
        self.camera_id = camera_id
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.stream_path = stream_path
        self.rtsp_id = rtsp_id
        self.rtsp_password = rtsp_password
        self.isRecording = False

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

    def disconnect(self):
        self.cap.release()
        
    def receive(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise NVRRecieveError("프레임 읽기 실패")

            if self.isRecording:
                self.out.write(frame)

            #TODO Kafka producer에 {camera_id, frame} 형태로 데이터 전달


    def startRecord(self):
        self.file_name = f"{uuid.uuid4()}.mp4"
        self.out = cv2.VideoWriter(self.file_name, self.fourcc, self.fps, (self.width, self.height))
        self.isRecording = True
    
    def endRecord(self):
        self.isRecording = False
        self.out.release()

        return self.file_name
    
class NVRClinet:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        db = PostgreSQL(os.getenv('DB_HOST'), os.getenv('DB_NAME'), os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_PORT'))
        camera_list = db.getCameraInfo()
        
        self.NVRChannelList = []
        for camera in camera_list:
            self.NVRChannelList.append(NVRChannel(**camera))

