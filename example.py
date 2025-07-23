import sys
import os
# poc 폴더를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc'))

try:
    from nvr_util.nvr_client import *
    from datetime import datetime
    
    # 1) 클라이언트 초기화
    client = NVRClient(name="rtsp")
    
    # 2) 채널 목록 조회
    channels = client.list_channels()
    for ch in channels:
        print(ch.id, ch.stream_url)
    
    # 3) Live Stream에서 프레임 읽기
    cap = client.get_live_stream(channel_id="1")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame → Detection-Tracking 모듈에 전달
    cap.release()
    
    # 4) 녹화영상에서 프레임 순회
    start = datetime(2025,7,23,9,55,0)
    end = datetime(2025,7,23,10,0,0)
    for frame in client.capture_frames("1", start, end):
        # frame → Detection-Tracking 모듈
        frame.show()

except NVRConnectionError as e:
    print(f"NVR 연결 오류: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")