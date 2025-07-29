import os
import cv2
import json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from kafka import KafkaConsumer
from dotenv import load_dotenv
from tracking_module import tracker

class FrameConsumer:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('TOPIC')
        '''
        fetch_min_bytes, fetch_max_wait_ms: Broker로부터 poll당 최소/최대 바이트를 얼마나 받을지 조절

        max_poll_records: 한 번에 poll()으로 가져올 최대 메시지 수

        session_timeout_ms, heartbeat_interval_ms: Consumer Group 안정성(Heartbeat) 관련 타이밍

        security_protocol, sasl_mechanism, ssl_ 옵션*: 보안(SSL/SASL) 설정
        '''
        self.consumer = KafkaConsumer('camera-frames',
                                      bootstrap_servers = self.broker,
                                        auto_offset_reset = 'earliest',
                                        enable_auto_commit = True,
                                        group_id = 'detection-tracker',
                                        key_deserializer = lambda b: b.decode('utf-8'),
                                        value_deserializer=lambda b: b,
                                        consumer_timeout_ms = None # ms
                                        )
        
    def get_msg():
        try:
            for msg in self.consumer:
                camera_id = msg.key
                jpeg_bytes = msg.value

                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                # 성공적으로 받은 메시지 처리
                    yield {
                        'camera_id': camera_id,
                        'frame': frame,
                        'timestamp': msg.timestamp,
                        'partition': msg.partition,
                        'offset': msg.offset
                    }
                else:
                    print(f"Failed to decode frame from camera {camera_id}")

        except Exception as e:
            print("Error: ",e)
            return e


class KafkaWorker:
    """
    Kafka I/O interface: consumes raw frames, uses DetectorTracker,
    and produces detection/tracking results to an output topic.
    """
    def __init__(self, engine_path: str):
        load_dotenv('env/aws.env')
        # input and output topics from env
        self.broker = os.getenv('BROKER')
        self.in_topic = os.getenv('INPUT_TOPIC')
        self.out_topic = os.getenv('OUTPUT_TOPIC')

        # Kafka consumer for frames
        self.consumer = KafkaConsumer(
            self.in_topic,
            bootstrap_servers=[self.broker],
            value_deserializer=lambda v: v,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='det-track-group'
        )
        # Kafka producer for results
        self.producer = KafkaProducer(
            bootstrap_servers=[self.broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        # Detector+Tracker
        self.detector_tracker = tracker.DetectorTracker(engine_path)