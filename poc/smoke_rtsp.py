import os, time, cv2, numpy as np
from dotenv import load_dotenv
from nvr_util import nvr_client

def load_env():
    for p in (os.getenv("DOTENV_PATH"),
              "/home/hiperwall/Ai_modules/Ai/env/aws.env",
              "env/aws.env", ".env"):
        if p and os.path.exists(p):
            load_dotenv(p, override=False)

def build_rtsp_url():
    # 1) RTSP_URL이 있으면 그대로 사용
    url = os.getenv("RTSP_URL")
    if url:
        return url

    # 2) 조합형 (Hikvision 예시: /Streaming/Channels/101)
    cam_ip   = os.getenv("CAMERA_IP")
    cam_port = os.getenv("CAMERA_PORT", "554")
    path     = os.getenv("STREAM_PATH", "/Streaming/Channels/101")
    user     = os.getenv("RTSP_ID", "admin")
    pw       = os.getenv("RTSP_PASSWORD", "")
    if not cam_ip:
        raise SystemExit("RTSP_URL 또는 CAMERA_IP가 필요합니다.")
    return f"rtsp://{user}:{pw}@{cam_ip}:{cam_port}{path}"

def configure_opencv(rtsp_over_tcp: bool):
    """
    OpenCV FFMPEG 캡처 옵션:
    - rtsp_transport: tcp/udp
    - stimeout: 소켓 오픈 타임아웃 (마이크로초)
    - max_delay: 디코더 최대 지연 (마이크로초)
    """
    opts = []
    opts.append(f"rtsp_transport;{'tcp' if rtsp_over_tcp else 'udp'}")
    # 5초 오픈 타임아웃
    opts.append("stimeout;5000000")
    # 2초 디코드 지연 제한
    opts.append("max_delay;2000000")
    # 버퍼링 억제
    opts.append("buffer_size;102400")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

def main():
    client = nvr_client.NVRClient()
    channel = client.NVRChannelList[0]
    channel.connect()

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

    cnt, t1 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[RTSP] read failed (stream hiccup). 재시도 중...")
                time.sleep(0.05)
                continue

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

if __name__ == "__main__":
    main()
