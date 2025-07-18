# nvr_util/nvr_client.py

import cv2
from datetime import datetime
from typing import List, Iterator
from onvif import ONVIFCamera
import subprocess
import shlex

from nvr_util.config import NVR_ENDPOINTS
from nvr_util.exceptions import (
    NVRError, NVRConnectionError, NVRAuthError, NVRChannelNotFoundError
)

class NVRChannel:
    def __init__(self, id: str, name: str, stream_url: str):
        self.id = id
        self.name = name
        self.stream_url = stream_url

class NVRClient:
    def __init__(self, name: str):
        cfg = next((e for e in NVR_ENDPOINTS if e["name"] == name), None)
        if not cfg:
            raise NVRConnectionError(f"NVR endpoint '{name}' not found")
        self.host     = cfg["host"]
        self.port     = cfg["port"]
        self.username = cfg["username"]
        self.password = cfg["password"]
        self.protocol = cfg["protocol"]    # "RTSP" or "ONVIF"
        self.channels = cfg["channels"]

    def list_channels(self) -> List[NVRChannel]:
        channel_list = []
        for ch in self.channels:
            if self.protocol.upper() == "RTSP":
                url = (
                    f"rtsp://{self.username}:{self.password}"
                    f"@{self.host}:{self.port}"
                    f"/cam/realmonitor?channel={ch}&subtype=0"
                )
            else:  # ONVIF
                cam = ONVIFCamera(self.host, self.port, self.username, self.password)
                media = cam.create_media_service()
                profile = media.GetProfiles()[int(ch)-1]
                url = media.GetStreamUri({
                    'StreamSetup': {
                        'Stream': 'RTP-Unicast',
                        'Transport': {'Protocol': 'RTSP'}
                    },
                    'ProfileToken': profile.token
                }).Uri

            channel_list.append(NVRChannel(id=ch, name=f"Channel-{ch}", stream_url=url))
        return channel_list

    def _get_channel(self, channel_id: str) -> NVRChannel:
        ch = next((c for c in self.list_channels() if c.id == channel_id), None)
        if not ch:
            raise NVRChannelNotFoundError(f"Channel {channel_id} not found")
        return ch

    def get_live_stream(self, channel_id: str) -> cv2.VideoCapture:
        channel = self._get_channel(channel_id)
        cap = cv2.VideoCapture(channel.stream_url)
        if not cap.isOpened():
            raise NVRConnectionError(f"Failed to open live stream: {channel.stream_url}")
        return cap

    def capture_frames(
        self,
        channel_id: str,
        start: datetime,
        end: datetime
    ) -> Iterator:
        """
        녹화된 영상의 start→end 구간 프레임을 순차적으로 yield.
        RTSP Playback URL: /cam/playback?channel={ch}&starttime={ISO}&endtime={ISO}&subtype=0
        """
        channel = self._get_channel(channel_id)

        # ISO 8601 UTC 문자열 생성 (예: '2025-07-18T10:00:00Z')
        def to_iso_z(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        start_iso = to_iso_z(start)
        end_iso   = to_iso_z(end)

        # Playback URL 구성
        playback_url = (
            f"rtsp://{self.username}:{self.password}"
            f"@{self.host}:{self.port}"
            f"/cam/playback?"
            f"channel={channel_id}&"
            f"starttime={start_iso}&"
            f"endtime={end_iso}&"
            f"subtype=0"
        )

        # OpenCV로 RTSP 재생
        cap = cv2.VideoCapture(playback_url)
        if not cap.isOpened():
            # 필요 시 FFMPEG subprocess 대안 예시
            raise NVRConnectionError(f"Failed to open playback stream: {playback_url}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def capture_snapshot(self, channel_id: str, timestamp: datetime):
        """
        지정 시각(timestamp)의 스냅샷을 한 장 리턴.
        - Live 스트림에서 가장 가까운 시점 frame을 캡처합니다.
        """
        cap = self.get_live_stream(channel_id)
        # TODO: 정확한 시각 seek 기능은 NVR SDK 또는 FFMPEG 사용을 권장
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise NVRError("Failed to capture snapshot")
        return frame
