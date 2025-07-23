# nvr_util/nvr_client.py (개선된 버전)

import cv2
from datetime import datetime
from typing import List, Iterator, Optional
import subprocess
import shlex
import logging

# 위에서 작성한 ONVIF 모듈 import
from onvif import ONVIFCamera

from nvr_util.config import NVR_ENDPOINTS
from nvr_util.exceptions import (
    NVRError, NVRConnectionError, NVRAuthError, NVRChannelNotFoundError
)

# 로깅 설정
logger = logging.getLogger(__name__)

class NVRChannel:
    def __init__(self, id: str, name: str, stream_url: str, profile_token: Optional[str] = None):
        self.id = id
        self.name = name
        self.stream_url = stream_url
        self.profile_token = profile_token  # ONVIF 프로필 토큰

class NVRClient:
    def __init__(self, name: str):
        cfg = next((e for e in NVR_ENDPOINTS if e["name"] == name), None)
        if not cfg:
            raise NVRConnectionError(f"NVR endpoint '{name}' not found")
        
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.username = cfg["username"]
        self.password = cfg["password"]
        self.protocol = cfg["protocol"]    # "RTSP" or "ONVIF"
        self.channels = cfg["channels"]
        
        # ONVIF 연결 캐싱
        self._onvif_camera = None

    def _get_onvif_camera(self) -> ONVIFCamera:
        """ONVIF 카메라 객체 생성 및 캐싱"""
        if self._onvif_camera is None:
            try:
                self._onvif_camera = ONVIFCamera(
                    self.host, self.port, self.username, self.password
                )
                logger.info(f"ONVIF 카메라 연결 성공: {self.host}:{self.port}")
            except Exception as e:
                raise NVRConnectionError(f"ONVIF 카메라 연결 실패: {str(e)}")
        
        return self._onvif_camera

    def list_channels(self) -> List[NVRChannel]:
        channel_list = []
        
        if self.protocol.upper() == "RTSP":
            # RTSP 프로토콜 처리
            for ch in self.channels:
                url = (
                    f"rtsp://{self.username}:{self.password}"
                    f"@{self.host}:{self.port}"
                    f"/cam/realmonitor?channel={ch}&subtype=0"
                )
                channel_list.append(NVRChannel(id=ch, name=f"Channel-{ch}", stream_url=url))
        
        elif self.protocol.upper() == "ONVIF":
            # ONVIF 프로토콜 처리
            try:
                onvif_camera = self._get_onvif_camera()
                profiles = onvif_camera.get_profiles()
                
                # 설정된 채널 수만큼 프로필 사용
                for i, ch in enumerate(self.channels):
                    if i < len(profiles):
                        profile = profiles[i]
                        try:
                            # 스트림 URI 조회
                            stream_uri = onvif_camera.get_stream_uri(profile.token)
                            
                            channel_list.append(NVRChannel(
                                id=ch,
                                name=f"Channel-{ch}",
                                stream_url=stream_uri,
                                profile_token=profile.token
                            ))
                            
                        except Exception as e:
                            logger.warning(f"채널 {ch} 스트림 URI 조회 실패: {str(e)}")
                            # 기본 RTSP URL로 폴백
                            fallback_url = f"rtsp://{self.username}:{self.password}@{self.host}:554/cam/realmonitor?channel={ch}&subtype=0"
                            channel_list.append(NVRChannel(id=ch, name=f"Channel-{ch}", stream_url=fallback_url))
                    else:
                        logger.warning(f"채널 {ch}에 대응하는 ONVIF 프로필이 없습니다.")
                        
            except Exception as e:
                logger.error(f"ONVIF 채널 목록 조회 실패: {str(e)}")
                raise NVRConnectionError(f"ONVIF 채널 목록 조회 실패: {str(e)}")
        
        else:
            raise ValueError(f"지원하지 않는 프로토콜: {self.protocol}")
            
        return channel_list

    def _get_channel(self, channel_id: str) -> NVRChannel:
        ch = next((c for c in self.list_channels() if c.id == channel_id), None)
        if not ch:
            raise NVRChannelNotFoundError(f"Channel {channel_id} not found")
        return ch

    def get_live_stream(self, channel_id: str) -> cv2.VideoCapture:
        channel = self._get_channel(channel_id)
        
        # OpenCV 설정 최적화
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        cap.set(cv2.CAP_PROP_FPS, 25)        # FPS 설정
        
        # RTSP 연결 시도
        success = cap.open(channel.stream_url, cv2.CAP_FFMPEG)
        if not success or not cap.isOpened():
            raise NVRConnectionError(f"Failed to open live stream: {channel.stream_url}")
        
        logger.info(f"라이브 스트림 연결 성공: 채널 {channel_id}")
        return cap

    def capture_frames(
        self,
        channel_id: str,
        start: datetime,
        end: datetime
    ) -> Iterator:
        """
        녹화된 영상의 start→end 구간 프레임을 순차적으로 yield
        """
        channel = self._get_channel(channel_id)

        # ISO 8601 UTC 문자열 생성
        def to_iso_z(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        start_iso = to_iso_z(start)
        end_iso = to_iso_z(end)

        # Playback URL 구성
        if self.protocol.upper() == "RTSP":
            playback_url = (
                f"rtsp://{self.username}:{self.password}"
                f"@{self.host}:{self.port}"
                f"/cam/playback?"
                f"channel={channel_id}&"
                f"starttime={start_iso}&"
                f"endtime={end_iso}&"
                f"subtype=0"
            )
        else:
            # ONVIF의 경우 라이브 스트림 URL 사용 (playback 구현은 복잡함)
            playback_url = channel.stream_url

        # OpenCV로 재생
        cap = cv2.VideoCapture(playback_url)
        if not cap.isOpened():
            raise NVRConnectionError(f"Failed to open playback stream: {playback_url}")

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                yield frame
            
            logger.info(f"총 {frame_count}개 프레임 처리 완료")
            
        except Exception as e:
            logger.error(f"프레임 캡처 중 오류: {str(e)}")
            raise NVRError(f"프레임 캡처 실패: {str(e)}")
        finally:
            cap.release()

    def capture_snapshot(self, channel_id: str, timestamp: datetime = None):
        """지정 시각의 스냅샷 캡처"""
        cap = self.get_live_stream(channel_id)
        
        try:
            # 몇 프레임 건너뛰어 안정화
            for _ in range(5):
                cap.read()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise NVRError("Failed to capture snapshot")
            
            logger.info(f"스냅샷 캡처 성공: 채널 {channel_id}")
            return frame
            
        finally:
            cap.release()

    def get_device_info(self) -> dict:
        """장치 정보 조회 (ONVIF만 지원)"""
        if self.protocol.upper() != "ONVIF":
            raise NotImplementedError("장치 정보 조회는 ONVIF 프로토콜만 지원합니다.")
        
        onvif_camera = self._get_onvif_camera()
        return onvif_camera.get_device_information()

    def control_ptz(self, channel_id: str, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0):
        """PTZ 제어 (ONVIF만 지원)"""
        if self.protocol.upper() != "ONVIF":
            raise NotImplementedError("PTZ 제어는 ONVIF 프로토콜만 지원합니다.")
        
        channel = self._get_channel(channel_id)
        if not channel.profile_token:
            raise NVRError(f"채널 {channel_id}의 프로필 토큰이 없습니다.")
        
        onvif_camera = self._get_onvif_camera()
        onvif_camera.move_ptz(channel.profile_token, pan, tilt, zoom)
        
        logger.info(f"PTZ 제어 실행: 채널 {channel_id}, pan={pan}, tilt={tilt}, zoom={zoom}")

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            if self.protocol.upper() == "ONVIF":
                onvif_camera = self._get_onvif_camera()
                return onvif_camera.test_connection()
            else:
                # RTSP는 채널 목록 조회로 테스트
                channels = self.list_channels()
                return len(channels) > 0
        except:
            return False