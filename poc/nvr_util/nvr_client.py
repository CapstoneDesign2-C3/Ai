    # nvr_util/nvr_client.py

import cv2
from datetime import datetime
from typing import List, Iterator
from onvif import ONVIFCamera    # ONVIF SDK 활용 시
from nvr_util.config import NVR_ENDPOINTS
from nvr_util.exceptions import (
    NVRConnectionError, NVRAuthError, NVRChannelNotFoundError
)

class NVRChannel:
    """NVR 카메라 채널 정보 객체."""
    def __init__(self, id: str, name: str, stream_url: str):
        self.id = id
        self.name = name
        self.stream_url = stream_url

class NVRClient:
    """
    NVR 스트림 및 녹화영상 접근을 추상화하는 Client.
    
    Attributes:
        host (str), port (int), username (str), password (str), protocol (str)
    """
    def __init__(self, name: str):
        cfg = next((e for e in NVR_ENDPOINTS if e["name"] == name), None)
        if not cfg:
            raise NVRConnectionError(f"NVR endpoint '{name}' not found")
        
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.username = cfg["username"]
        self.password = cfg["password"]
        self.protocol = cfg["protocol"]  # "RTSP" or "ONVIF"
        self.channels = cfg["channels"]

    def list_channels(self) -> List[NVRChannel]:
        """
        등록된 채널 목록 반환.
        ONVIF: GetProfiles API 사용
        RTSP: config 내 channels 정보 사용
        """
        channel_list = []
        for ch in self.channels:
            if self.protocol == "RTSP":
                url = f"rtsp://{self.username}:{self.password}@{self.host}:{self.port}/cam/realmonitor?channel={ch}&subtype=0"
            else:  # ONVIF
                cam = ONVIFCamera(self.host, self.port, self.username, self.password)
                profile = cam.create_media_service().GetProfiles()[int(ch)-1]
                url = cam.create_media_service().GetStreamUri({
                    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                    'ProfileToken': profile.token
                }).Uri

            channel_list.append(NVRChannel(id=ch, name=f"Channel-{ch}", stream_url=url))
        return channel_list

    def get_live_stream(self, channel_id: str) -> cv2.VideoCapture:
        """
        채널의 Live Stream(VideoCapture) 객체 반환.
        """
        channel = next((c for c in self.list_channels() if c.id == channel_id), None)
        if not channel:
            raise NVRChannelNotFoundError(f"Channel {channel_id} not found")
        cap = cv2.VideoCapture(channel.stream_url)   # OpenCV VideoCapture 사용
        if not cap.isOpened():
            raise NVRConnectionError(f"Failed to open stream: {channel.stream_url}")
        return cap

    def capture_frames(self,
                       channel_id: str,
                       start: datetime,
                       end: datetime) -> Iterator:
        """
        특정 시간대(start→end)의 녹화 영상에서 프레임 단위로 yield.
        * 구현 예시: FFMPEG subprocess 호출 또는 NVR SDK API 활용
        """
        # TODO: 실제 recorder retrieval 구현
        # ex) ffmpeg -i "rtsp://.../record?start=...&end=..." -f image2pipe ...
        raise NotImplementedError

    def capture_snapshot(self, channel_id: str, timestamp: datetime):
        """
        지정 시각의 스냅샷(Frame) 반환.
        """
        cap = self.get_live_stream(channel_id)
        # TODO: 타임시크 기능 있는 SDK 활용 권장
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise NVRError("Failed to capture snapshot")
        return frame
