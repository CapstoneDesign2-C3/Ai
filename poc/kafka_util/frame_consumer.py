from kafka import KafkaConsumer
from dotenv import load_dotenv
import os

class FrameConsumer:
    def __init__(self):
        dotenv_path = 'env/aws.env'
        load_dotenv(dotenv_path)
        self.broker = os.getenv('BROKER')
        self.topic = os.getnev('TOPIC')
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
                                        consumer_timeout_ms = None # ms
                                        )
        
        def get_msg():
            try:
                for msg in self.consumer:

            except Exception as e:
                print("Error: ",e)
                return e
