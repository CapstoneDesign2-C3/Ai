import sys
import os
# poc 폴더를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc'))
from nvr_util.nvr_client import *
from datetime import datetime, timedelta

# HTTP NVR 클라이언트 생성
nvr = NVRClient("http_nvr")

# 특정 시간 범위의 영상 스트리밍
start_time = datetime(2025, 7, 23, 10, 0, 0)
end_time = datetime(2025, 7, 23, 11, 0, 0)

# 프레임별로 처리
for frame in nvr.capture_frames("1", start_time, end_time):
    # 프레임 처리 로직
    cv2.imshow("Playback", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()