from kafka import KafkaConsumer
from deep_sort_realtime.deepsort_tracker import DeepSort
from poc.kafka_util import consumers
import cv2
import pickle  # 예시: 메시지 디코딩

def main(camera_id):
    # 1) Kafka Consumer: camera_id(key) → 파티션 고정
    consumer = consumers.FrameConsumer().consumer

    # 2) 이 프로세스는 오직 camera_id 파티션만 할당받음
    #    (consumer.subscribe() + rebalance listener 로 확인 가능)
    consumer.subscribe()
    
    # 3) Tracker 인스턴스 생성 (카메라별 독립)
    tracker = DeepSort(max_age=5)

    for msg in consumer:
        frame = msg.value['frame']        # 예: {'frame': numpy.ndarray, …}
        bbs   = msg.value['detections']   # 예: [([l,t,w,h], conf, cls), …]

        # 4) 추적 업데이트
        tracks = tracker.update_tracks(bbs, frame=frame)

        # 5) 확정된 트랙만 사용
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            # → 이 시점에 DB에 저장하거나, downstream 모듈로 전달

if __name__ == '__main__':
    import sys
    # argv[1]에 camera_id를 넣어서 띄움
    main(camera_id=int(sys.argv[1]))
