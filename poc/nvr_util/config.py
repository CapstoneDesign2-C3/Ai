# nvr_util/config.py

from typing import List, Dict
from datetime import datetime

NVR_ENDPOINTS: List[Dict] = [
    {
        "name": "test_nvr",
        "host": "192.168.5.99",
        "port": 554,
        "username": "Admin",
        "password": "hiperwall2018",
        "protocol": "RTSP",       # or "ONVIF"
        "channels": ["1", "2", "3", "4"]
    },
    # 필요한 만큼 추가
]
