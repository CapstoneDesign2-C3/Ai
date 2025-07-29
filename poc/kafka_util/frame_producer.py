from kafka import KafkaProducer
from dotenv import load_dotenv
import json
import cv2
import os

class FrameProducer:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('FRAME_TOPIC')
        self.producer = KafkaProducer(bootstrap_servers=self.broker,
                                      value_serializer=lambda x: x,
                                      acks=0,
                                      api_version=(2,5,0),
                                      retries=3
                                      )

    # pre-partitioning action
    # frame, cameraid merge to json
    # json serialization
    def send_message(self, cameraID, frame):
        try:
            # encode frame to jpg
            _, buffer = cv2.imencode('.jpg', frame)
            jpeg_bytes = buffer.tobytes()

            self.producer.send(
            self.topic,
            key=str(cameraID).encode('utf-8'),
            value=jpeg_bytes
            )
            self.producer.flush()   # 비우는 작업
            print(f"[Kafka] Sent frame from camera {cameraID} to topic {self.topic}")
            return {'status_code': 200, 'error': None}
        except Exception as e:
            print("Error: Sending Error", e)
            return e
    


'''
# example

import time
from csv import reader

# 브로커와 토픽명을 지정한다.
broker = 'localhost:9092'
topic = 'new-topic'
message_producer = MessageProducer(broker, topic)

with open('test/'+topic+'.txt', 'r', encoding='utf-8') as file:
    for data in file:
        print("send-data: ", data)
        res = message_producer.send_message(data)
'''