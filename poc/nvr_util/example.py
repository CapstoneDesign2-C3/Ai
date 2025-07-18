from nvr_util.nvr_client import NVRClient
from datetime import datetime

# 1) 클라이언트 초기화
client = NVRClient(name="MainNVR")

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

# 4) 녹화영상(예: 2025-07-18 10:00~10:05)에서 프레임 순회
start = datetime(2025,7,18,10,0,0)
end   = datetime(2025,7,18,10,5,0)
for frame in client.capture_frames("1", start, end):
    # frame → Detection-Tracking 모듈
