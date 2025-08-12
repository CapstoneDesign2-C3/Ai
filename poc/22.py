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
    client = nvr_client.NVRClient()
    channel = client.NVRChannelList[1]
    channel.connect()
    camera_id = channel.camera_id
    # FFMPEG 백엔드로 강제
    cap = channel.cap

    if not cap.isOpened():
        print("[RTSP] open failed. 다시 UDP/TCP 바꿔보거나 ffplay로 확인해보세요.")
        return

    # 버퍼 사이즈 줄이기 (가능한 빌드에서만 영향)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 첫 프레임 대기
    t0 = time.time()
    for i in range(100):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.02)
    else:
        print("[RTSP] no frame within timeout (2s). 경로/권한/채널 확인 필수.")
        cap.release()
        return

    first_latency_ms = int((time.time() - t0) * 1000)
    h, w = frame.shape[:2]
    cv2.imwrite("rtsp_first_frame.jpg", frame)
    print(f"[RTSP] first frame: {w}x{h}, first_latency={first_latency_ms} ms (saved: rtsp_first_frame.jpg)")

    # FPS 측정 + 화면 표시
    win = "RTSP Preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    detector = DetectorAndTracker(cameraID=camera_id)

    cnt, t1 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[RTSP] read failed (stream hiccup). 재시도 중...")
                time.sleep(0.05)
                continue

            # c) 처리 시간 출력
            vis, timing_info, boxes, scores, class_ids, tracks = detector.detect_and_track(frame, debug=False, return_vis=True)
            ms = {k: timing_info[k]*1000.0 for k in ('parse','nms','inference','postprocess','total')}
            y = 30
            for k in ('total','inference','parse','nms','postprocess'):
                cv2.putText(vis, f"{k}: {ms[k]:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y += 28

            # d) 결과 디스플레이
            cv2.imshow('Detection & Tracking', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[RTSP] closed.")

if __name__ == "__main__":
    main()
