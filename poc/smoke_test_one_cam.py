import os, time, cv2, numpy as np
from dotenv import load_dotenv
from nvr_util import nvr_client
from tracking_module.detection_and_tracking import DetectorAndTracker

def load_env():
    for p in (os.getenv("DOTENV_PATH"),
              "/home/hiperwall/Ai_modules/Ai/env/aws.env",
              "env/aws.env", ".env"):
        if p and os.path.exists(p):
            load_dotenv(p, override=False)

def main():
    load_env()

    client = nvr_client.NVRClient()
    channel = client.NVRChannelList[1]
    channel.connect()
    camera_id = channel.camera_id
    cap = channel.cap  # OpenCV FFMPEG 백엔드

    if not cap.isOpened():
        print("[RTSP] open failed. 다시 UDP/TCP 바꿔보거나 ffplay로 확인해보세요.")
        return

    # 버퍼 사이즈 매우 작게(가능한 빌드에서만 영향)
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
        print("[RTSP] no frame within timeout (2s). 경로/권한/채널 확인 필수.")
        cap.release()
        return

    first_latency_ms = int((time.time() - t0) * 1000)
    h, w = frame.shape[:2]
    cv2.imwrite("rtsp_first_frame.jpg", frame)
    print(f"[RTSP] first frame: {w}x{h}, first_latency={first_latency_ms} ms (saved: rtsp_first_frame.jpg)")

    win = "RTSP Preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    detector = DetectorAndTracker(cameraID=camera_id, tracker_type='ocsort')

    # FPS 측정
    last_t = time.time()
    frame_cnt = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[RTSP] read failed (stream hiccup). 재시도 중...")
                time.sleep(0.05)
                continue

            # 탐지+추적(+시각화 프레임) 한 번에
            vis, timing_info, boxes, scores, class_ids, tracks = detector.detect_and_track(
                frame, debug=False, return_vis=True
            )

            # FPS 갱신
            frame_cnt += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = frame_cnt / (now - last_t)
                frame_cnt = 0
                last_t = now

            # 타이밍 오버레이(키 없으면 0으로)
            overlay_keys = ('total', 'inference', 'parse', 'nms', 'postprocess')
            ms = {}
            for k in overlay_keys:
                v = timing_info.get(k, 0.0) if isinstance(timing_info, dict) else 0.0
                ms[k] = float(v) * 1000.0
            y = 28
            for k in overlay_keys:
                cv2.putText(vis, f"{k}: {ms[k]:.1f} ms", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 26
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, y+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 화면 표시
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[RTSP] closed.")

if __name__ == "__main__":
    main()
