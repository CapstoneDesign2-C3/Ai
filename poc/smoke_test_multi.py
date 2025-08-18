import os, time, cv2, numpy as np
import threading
from dotenv import load_dotenv
from nvr_util import nvr_client
from tracking_module.detection_and_tracking import DetectorAndTracker
import logging
from math import ceil, sqrt
from collections import defaultdict

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s"
)
logging.getLogger("kafka.coordinator").setLevel(logging.INFO)  # 너무 시끄러우면 INFO

# ====== ENV LOADER ======
def load_env():
    for p in (os.getenv("DOTENV_PATH"),
              "/home/hiperwall/Ai_modules/Ai/env/aws.env",
              "env/aws.env", ".env"):
        if p and os.path.exists(p):
            load_dotenv(p, override=False)
            logging.info("dotenv loaded: %s", p)
            break

# ====== WORKER ======
def camera_worker(channel, camera_id, out_frames, sizes, stop_event, gpu_sema, tracker_type):
    log = logging.getLogger(f"cam-{camera_id}")

    # 채널 연결
    try:
        channel.connect()
    except Exception as e:
        log.exception("channel.connect() failed: %s", e)
        return

    cap = channel.cap
    if not cap or not cap.isOpened():
        log.error("[RTSP] open failed. 경로/권한/채널 확인 필수.")
        return

    # 버퍼 사이즈 작게
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 첫 프레임 대기(최대 ~2s)
    t0 = time.time()
    frame = None
    for _ in range(100):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
        time.sleep(0.02)
    if frame is None:
        log.error("[RTSP] no frame within timeout (2s).")
        cap.release()
        return

    first_latency_ms = int((time.time() - t0) * 1000)
    h, w = frame.shape[:2]
    sizes[camera_id] = (w, h)  # 메인 스레드에서 타일 크기 계산용
    log.info("[RTSP] first frame: %dx%d, first_latency=%d ms", w, h, first_latency_ms)

    # Detector 초기화
    try:
        detector = DetectorAndTracker(cameraID=camera_id, tracker_type=tracker_type)
    except Exception as e:
        log.exception("Detector init failed: %s", e)
        cap.release()
        return

    # FPS 측정
    last_t = time.time()
    frame_cnt = 0
    fps = 0.0

    # 재연결/에러 관리
    consecutive_fail = 0
    MAX_FAIL = 50

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_fail += 1
                if consecutive_fail > MAX_FAIL:
                    log.error("[RTSP] read failed %d times. stopping.", consecutive_fail)
                    break
                time.sleep(0.05)
                continue
            consecutive_fail = 0

            # 추론 동시성 제한 (GPU 안정성)
            with gpu_sema:
                try:
                    vis, timing_info, boxes, scores, class_ids, tracks = detector.detect_and_track(
                        frame, debug=False, return_vis=True
                    )
                except Exception as e:
                    log.exception("detect_and_track failed: %s", e)
                    continue

            # FPS 업데이트
            frame_cnt += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = frame_cnt / (now - last_t)
                frame_cnt = 0
                last_t = now

            # 타이밍 오버레이
            overlay_keys = ('total', 'inference', 'parse', 'nms', 'postprocess')
            ms = {k: float(timing_info.get(k, 0.0)) * 1000.0 if isinstance(timing_info, dict) else 0.0
                  for k in overlay_keys}
            y = 22
            for k in overlay_keys:
                cv2.putText(vis, f"{k}: {ms[k]:.1f} ms", (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 20
            cv2.putText(vis, f"FPS: {fps:.1f}", (8, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"CAM:{camera_id}", (8, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # 최신 프레임 저장(메인 스레드가 모자이크 표시)
            out_frames[camera_id] = vis

    finally:
        try:
            cap.release()
        except Exception:
            pass
        logging.info("[RTSP] closed for cam %s.", camera_id)

# ====== MOSAIC ======
def make_mosaic(frames_dict, order, tile_size):
    """
    frames_dict: {camera_id: BGR frame}
    order: list of camera_id to display order-stable
    tile_size: (tw, th)
    """
    if not order:
        return None
    tw, th = tile_size
    n = len(order)
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))
    mosaic = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for idx, cid in enumerate(order):
        r = idx // cols
        c = idx % cols
        x1, y1 = c * tw, r * th
        frame = frames_dict.get(cid)
        if frame is None:
            # placeholder
            cv2.putText(mosaic, f"CAM {cid}: no frame", (x1 + 10, y1 + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue
        h, w = frame.shape[:2]
        # 같은 종횡비로 맞춰 리사이즈
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        # 중앙 배치
        ox = x1 + (tw - nw) // 2
        oy = y1 + (th - nh) // 2
        mosaic[oy:oy + nh, ox:ox + nw] = resized

    return mosaic

# ====== MAIN ======
def main():
    load_env()

    client = nvr_client.NVRClient()
    channels = getattr(client, "NVRChannelList", None) or []
    if not channels:
        print("No channels found in NVRClient.NVRChannelList")
        return

    # 환경 변수로 동시에 몇 개 처리할지 제한(기본 전체)
    max_cams = int(os.getenv("MAX_CAMS", "0"))  # 0이면 전체
    tracker_type = os.getenv("TRACKER_TYPE", "ocsort")
    gpu_conc = int(os.getenv("GPU_CONCURRENCY", "1"))

    selected = channels if max_cams <= 0 else channels[:max_cams]
    cam_ids = []
    for ch in selected:
        try:
            cam_ids.append(ch.camera_id)
        except Exception:
            cam_ids.append(len(cam_ids))  # fallback

    stop_event = threading.Event()
    gpu_sema = threading.Semaphore(gpu_conc)

    # 공유 버퍼(스레드 안전 dict)
    out_frames = {}
    sizes = {}  # {camera_id: (w,h)}

    # 워커 스레드 기동
    threads = []
    for ch, cid in zip(selected, cam_ids):
        t = threading.Thread(
            target=camera_worker,
            args=(ch, cid, out_frames, sizes, stop_event, gpu_sema, tracker_type),
            daemon=True
        )
        t.start()
        threads.append(t)
        logging.info("Started camera thread: camID=%s", cid)

    # 모자이크 창
    win = "Multi-Cam Preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 타일 크기 결정: 첫 사이즈가 들어오면 그 비율로
    tile_w = int(os.getenv("TILE_W", "640"))
    tile_h = int(os.getenv("TILE_H", "360"))

    order = list(cam_ids)

    try:
        while True:
            # 종료 키
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            # 모자이크 만들기
            mosaic = make_mosaic(out_frames, order, (tile_w, tile_h))
            if mosaic is not None:
                cv2.imshow(win, mosaic)

            # 스레드가 하나라도 죽었는지 감시(선택)
            for t in threads:
                if not t.is_alive():
                    logging.warning("A camera thread has stopped. Press 'q' to exit.")
                    # 계속 보기 원하면 break/continue로 유지 가능

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)
        cv2.destroyAllWindows()
        logging.info("All stopped.")

if __name__ == "__main__":
    main()
