# run_nvr_kafka_e2e.py
import os, time, threading, cv2, numpy as np
from dotenv import load_dotenv
from nvr_util import nvr_client
from kafka_util.consumers import create_frame_consumer
from kafka_util.producers import create_frame_producer


def load_env():
    for p in (os.getenv("DOTENV_PATH"),
              "/home/hiperwall/Ai_modules/Ai/env/aws.env",
              "env/aws.env", ".env"):
        if p and os.path.exists(p):
            load_dotenv(p, override=False)

def start_consumer(camera_id: str):
    win = f"recv-{camera_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_frame(frame: np.ndarray, headers: dict):
        ts = headers.get("ts_ms")
        latency = None
        try:
            if ts: latency = int(time.time()*1000) - int(ts)
        except: pass
        cv2.putText(frame, f"{camera_id} | {latency or 'NA'} ms", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt()

    cons = create_frame_consumer(camera_id, on_frame)

    def run():
        try:
            cons.run()
        except KeyboardInterrupt:
            pass
        finally:
            try: cons.stop()
            except: pass
            cv2.destroyWindow(win)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    load_env()

    # 1) NVR 채널 선택 & RTSP 연결
    client = nvr_client.NVRClient()
    ch = client.NVRChannelList[0]   # 필요시 인덱스 조정
    ch.connect()                    # 여기서 cap 준비됨
    camera_id = ch.camera_id
    cap = ch.cap

    # 2) 컨슈머 시작 (Kafka → 화면 표시)
    t_cons = start_consumer(camera_id)
    prod = create_frame_producer(camera_id=camera_id)

    # 3) 프로듀서 시작 (RTSP → Kafka)
     # FPS 측정 + 화면 표시
    win = "RTSP Preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cnt, t1 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[RTSP] read failed (stream hiccup). 재시도 중...")
                time.sleep(0.05)
                continue
            
            prod.send_message(frame=frame)
            
            cnt += 1
            now = time.time()
            
            if now - t1 >= 1.0:
                fps = cnt / (now - t1)
                print(f"[RTSP] recv fps ~ {fps:.1f}")
                cnt, t1 = 0, now

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[RTSP] closed.")
