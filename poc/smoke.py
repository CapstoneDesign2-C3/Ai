# smoke_kafka_frames.py
import numpy as np, cv2, time, os
from kafka_util.producers import create_frame_producer
from kafka_util.consumers import create_frame_consumer

cam = os.getenv("CAMERA_ID", "cam-001")

def on_frame(frame, headers):
    print("[OK] got frame", headers)
    cv2.imshow("smoke", frame)
    cv2.waitKey(500)
    raise KeyboardInterrupt()

# consumer 먼저 대기
import threading
cons = create_frame_consumer(cam, on_frame)
t = threading.Thread(target=cons.run, daemon=True); t.start()

# 200ms 대기 후 1장 전송
time.sleep(0.2)
prod = create_frame_producer(cam)
img = np.full((480, 640, 3), 128, np.uint8)
cv2.putText(img, "SMOKE", (180,240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
prod.send_message(img, quality=85)
print("[SMOKE] sent one frame")
time.sleep(2)
