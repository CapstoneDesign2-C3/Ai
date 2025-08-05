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
        self.topic = os.getenv('FRAME_TOPIC')
        self.consumer = KafkaConsumer(self.topic,
                                      bootstrap_servers = self.broker,
                                        auto_offset_reset = 'earliest',
                                        enable_auto_commit = True,
                                        group_id = 'frame_consumer',
                                        key_deserializer = lambda b: b.decode('utf-8'),
                                        value_deserializer=lambda b: b,
                                        consumer_timeout_ms = None # ms
                                        )

class DetectedResultConsumer:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('DETECTED_RESULT')
        self.consumer = KafkaConsumer(self.topic,
                                        bootstrap_servers=self.broker,
                                        auto_offset_reset = 'earliest',
                                        enable_auto_commit = True,
                                        group_id = 'result_consumer',
                                        key_deserializer = lambda b: b.decode('utf-8'),
                                        value_deserializer=lambda b: b,
                                        consumer_timeout_ms = None # ms
                                        )   