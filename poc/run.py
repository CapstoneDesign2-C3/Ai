import os, time, cv2, threading, signal, logging
from dotenv import load_dotenv
from nvr_util import nvr_client
from tracking_module.detection_and_tracking import DetectorAndTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
)
logging.getLogger("kafka.coordinator").setLevel(logging.INFO)

STOP = threading.Event()

def _prime_first_frame(cap, timeout_sec=2.0):
    t0 = time.time()
    frame = None
    while time.time() - t0 < timeout_sec and not STOP.is_set():
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
        time.sleep(0.02)
    return frame, int((time.time() - t0) * 1000)

def _setup_cap_latency(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def run_camera_loop(channel, tracker_type="ocsort"):
    """
    채널 하나당 스레드:
      - RTSP 연결
      - 프레임 읽기 실패 시 재시도/재연결
      - DetectorAndTracker로 추적
      - 시각화 없음(return_vis=False)
    """
    log = logging.getLogger(f"cam-{channel.camera_id}")
    reconnect_backoff = 1.0

    while not STOP.is_set():
        try:
            channel.connect()  # nvr_client.NVRChannel.connect()
            cap = channel.cap
            if not cap.isOpened():
                raise RuntimeError("OpenCV cap open failed")

            _setup_cap_latency(cap)
            frame, first_ms = _prime_first_frame(cap, timeout_sec=2.0)
            if frame is None:
                raise RuntimeError("No frame within 2s")

            h, w = frame.shape[:2]
            log.info("Connected: %dx%d first_latency=%dms", w, h, first_ms)

            detector = DetectorAndTracker(cameraID=channel.camera_id, tracker_type=tracker_type)

            fail = 0
            while not STOP.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail += 1
                    if fail >= 50:
                        raise RuntimeError("RTSP read fail threshold reached")
                    time.sleep(0.05)
                    continue
                fail = 0

                # 탐지 + 추적 (시각화 제거)
                _vis, _timing, _boxes, _scores, _cids, _tracks = detector.detect_and_track(
                    frame, debug=False, return_vis=False
                )

            # 정상 종료
            try:
                cap.release()
            except Exception:
                pass
            break

        except Exception as e:
            log.error("Loop error: %s", e)
            # 재연결 백오프
            try:
                if getattr(channel, "cap", None) is not None:
                    channel.cap.release()
            except Exception:
                pass
            if STOP.is_set():
                break
            time.sleep(reconnect_backoff)
            reconnect_backoff = min(reconnect_backoff * 2, 10.0)
            log.info("Reconnecting... backoff=%.1fs", reconnect_backoff)

def main():
    # NVRClient 내부에서 .env를 읽지만, 바깥에서 한 번 더 읽어도 무해
    tracker_type = os.getenv("TRACKER_TYPE", "ocsort").strip().lower()

    client = nvr_client.NVRClient()  # DB에서 camera_info 읽어 채널 리스트 생성
    channels = client.NVRChannelList
    if not channels:
        logging.error("No cameras found in DB camera_info.")
        return

    logging.info("Starting %d camera workers: %s", len(channels), [c.camera_id for c in channels])

    threads = []
    for ch in channels:
        t = threading.Thread(target=run_camera_loop, args=(ch, 'ocsort'), daemon=True)
        t.start()
        threads.append(t)

    def _sigterm_handler(signum, frame):
        logging.info("Received signal %s, stopping...", signum)
        STOP.set()

    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    finally:
        STOP.set()
        logging.info("All camera loops stopped.")

if __name__ == "__main__":
    main()
