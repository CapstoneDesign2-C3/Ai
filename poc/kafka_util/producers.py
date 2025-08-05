from kafka import KafkaProducer
from dotenv import load_dotenv
import json
import cv2
import os

class FrameProducer:
    def __init__(self, cameraID):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('FRAME_TOPIC')
        self.cameraID = cameraID
        self.producer = KafkaProducer(bootstrap_servers=self.broker,
                                      value_serializer=lambda x: x,
                                      acks=0,
                                      api_version=(2,5,0),
                                      retries=3
                                      )

    # pre-partitioning action
    # frame, cameraid merge to json
    # json serialization
    def send_message(self, frame):
        try:
            # encode frame to jpg
            _, buffer = cv2.imencode('.jpg', frame)
            jpeg_bytes = buffer.tobytes()

            self.producer.send(
            self.topic,
            key=str(self.cameraID).encode('utf-8'),
            value=jpeg_bytes
            )
            
            # 1frame 1flush is it ok?
            self.producer.flush()   # 비우는 작업
            return {'status_code': 200, 'error': None}
        except Exception as e:
            print("Error: Sending Error", e)
            return e
    
class DetectedResultProducer:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getenv('DETECTED_RESULT')
        self.producer = KafkaProducer(  bootstrap_servers=self.broker,
                                        key_serializer=lambda k: k.encode('utf-8'),
                                        value_serializer=lambda x: x,
                                        acks=0,
                                        api_version=(2,5,0),  
                                        retries=3
                                    )
    
    def send_message(self, cameraID, payload):
        self.producer.send( self.topic,
                            key=cameraID,
                            value=payload
                            )
