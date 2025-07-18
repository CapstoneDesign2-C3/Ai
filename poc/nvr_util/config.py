# nvr_util/config.py

from typing import List, Dict
from datetime import datetime

NVR_ENDPOINTS: List[Dict] = [
    {
        "name": "MainNVR",
        "host": "192.168.1.100",
        "port": 554,
        "username": "admin",
        "password": "password",
        "protocol": "RTSP",       # or "ONVIF"
        "channels": ["1", "2", "3", "4"]
    },
    # 필요한 만큼 추가
]
