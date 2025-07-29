import json
import time
from kafka import KafkaConsumer
from dotenv import load_dotenv
import os
import threading

# 환경 변수 로드
load_dotenv('env/aws.env')
BROKER = os.getenv('BROKER', 'localhost:9092')

def monitor_topic(topic_name, max_messages=10):
    """특정 토픽의 메시지를 모니터링"""
    print(f"[Monitor] Starting monitoring for topic: {topic_name}")
    
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[BROKER],
            value_deserializer=lambda x: x,
            auto_offset_reset='latest',  # 최신 메시지부터
            group_id=f'monitor_{topic_name}',
            consumer_timeout_ms=5000
        )
        
        message_count = 0
        for msg in consumer:
            message_count += 1
            
            try:
                # JSON 메시지인 경우
                if topic_name in ['detection-results', 'reid-requests', 'reid-responses']:
                    data = json.loads(msg.value.decode('utf-8'))
                    if topic_name == 'detection-results':
                        tracks = data.get('tracks', [])
                        print(f"[{topic_name}] Camera {data.get('camera_id')}: {len(tracks)} tracks")
                        if tracks:
                            print(f"  Sample track: ID={tracks[0].get('local_id')}, bbox={tracks[0].get('bbox')}")
                            
                    elif topic_name == 'reid-requests':
                        print(f"[{topic_name}] Camera {data.get('camera_id')}, Local ID: {data.get('local_id')}")
                        
                    elif topic_name == 'reid-responses':
                        print(f"[{topic_name}] Camera {data.get('camera_id')}, Local ID: {data.get('local_id')} -> Global ID: {data.get('global_id')}")
                        
                # 이미지 데이터인 경우
                else:
                    key = msg.key.decode('utf-8') if msg.key else 'unknown'
                    size = len(msg.value)
                    print(f"[{topic_name}] Key: {key}, Size: {size} bytes")
                    
            except Exception as e:
                print(f"[{topic_name}] Message parsing error: {e}")
            
            if message_count >= max_messages:
                print(f"[Monitor] Reached max messages ({max_messages}) for {topic_name}")
                break
                
        consumer.close()
        
    except Exception as e:
        print(f"[Monitor] Error monitoring {topic_name}: {e}")

def main():
    """모든 관련 토픽들을 모니터링"""
    topics = [
        'camera-frames',
        'detection-results', 
        'reid-requests',
        'reid-responses'
    ]
    
    print(f"[*] Starting Kafka topic monitor for broker: {BROKER}")
    print(f"[*] Monitoring topics: {', '.join(topics)}")
    
    # 각 토픽을 별도 스레드에서 모니터링
    threads = []
    for topic in topics:
        thread = threading.Thread(target=monitor_topic, args=(topic, 20))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        time.sleep(1)  # 스레드 시작 간격
    
    try:
        # 메인 스레드에서 대기
        while True:
            time.sleep(10)
            print(f"[*] Monitor running... (threads active: {sum(1 for t in threads if t.is_alive())})")
            
    except KeyboardInterrupt:
        print("\n[*] Stopping monitor...")

if __name__ == '__main__':
    main()