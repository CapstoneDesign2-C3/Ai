# nvr_util/nvr_client.py (HTTP 지원 개선 버전)

import cv2
import requests
import json
from datetime import datetime
from typing import List, Iterator, Optional, Dict
import subprocess
import shlex
import logging
import time
from urllib.parse import urlencode
import base64

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
        self.protocol = cfg["protocol"]    # "RTSP", "ONVIF", "HTTP"
        self.channels = cfg["channels"]
        
        # HTTP API 엔드포인트 설정 (제조사별로 다름)
        self.api_base = cfg.get("api_base", "/api/v1")  # API 베이스 경로
        self.stream_path = cfg.get("stream_path", "/stream")  # 스트림 경로
        
        # ONVIF 연결 캐싱
        self._onvif_camera = None
        
        # HTTP 세션 설정
        self._http_session = None
        if self.protocol.upper() == "HTTP":
            self._setup_http_session()

    def _setup_http_session(self):
        """HTTP 세션 설정 및 인증"""
        self._http_session = requests.Session()
        
        # Basic Auth 설정
        auth_str = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        self._http_session.headers.update({
            'Authorization': f'Basic {auth_str}',
            'Content-Type': 'application/json',
            'User-Agent': 'NVR-Client/1.0'
        })
        
        # 연결 테스트
        try:
            response = self._http_session.get(
                f"http://{self.host}:{self.port}{self.api_base}/status",
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"HTTP NVR 연결 성공: {self.host}:{self.port}")
            else:
                logger.warning(f"HTTP NVR 연결 확인 불가: {response.status_code}")
        except Exception as e:
            logger.warning(f"HTTP NVR 연결 테스트 실패: {str(e)}")

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

    def _make_http_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> requests.Response:
        """HTTP 요청 수행"""
        if not self._http_session:
            raise NVRConnectionError("HTTP 세션이 초기화되지 않았습니다.")
        
        url = f"http://{self.host}:{self.port}{self.api_base}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self._http_session.get(url, timeout=30)
            elif method.upper() == "POST":
                response = self._http_session.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메소드: {method}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            raise NVRConnectionError(f"HTTP 요청 실패: {str(e)}")

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
        
        elif self.protocol.upper() == "HTTP":
            # HTTP 프로토콜 처리
            try:
                # HTTP API를 통해 채널 목록 조회
                response = self._make_http_request("/channels")
                channels_data = response.json()
                
                # API 응답 형식에 따라 파싱 (제조사별로 다를 수 있음)
                if isinstance(channels_data, list):
                    api_channels = channels_data
                elif isinstance(channels_data, dict) and "channels" in channels_data:
                    api_channels = channels_data["channels"]
                else:
                    # 설정 파일의 채널 목록 사용
                    api_channels = [{"id": ch, "name": f"Channel-{ch}"} for ch in self.channels]
                
                for ch_info in api_channels:
                    ch_id = str(ch_info.get("id", ch_info.get("channel", "")))
                    ch_name = ch_info.get("name", f"Channel-{ch_id}")
                    
                    # HTTP 스트림 URL 구성
                    stream_url = (
                        f"http://{self.username}:{self.password}@{self.host}:{self.port}"
                        f"{self.stream_path}/channel/{ch_id}/live"
                    )
                    
                    channel_list.append(NVRChannel(id=ch_id, name=ch_name, stream_url=stream_url))
                    
            except Exception as e:
                logger.warning(f"HTTP API 채널 조회 실패, 설정 파일 사용: {str(e)}")
                # 설정 파일의 채널 목록으로 폴백
                for ch in self.channels:
                    stream_url = (
                        f"http://{self.username}:{self.password}@{self.host}:{self.port}"
                        f"{self.stream_path}/channel/{ch}/live"
                    )
                    channel_list.append(NVRChannel(id=ch, name=f"Channel-{ch}", stream_url=stream_url))
            
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
        
        # 프로토콜별 연결
        if self.protocol.upper() == "HTTP":
            # HTTP 스트리밍의 경우 FFmpeg 백엔드 사용
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        success = cap.open(channel.stream_url, cv2.CAP_FFMPEG)
        if not success or not cap.isOpened():
            raise NVRConnectionError(f"Failed to open live stream: {channel.stream_url}")
        
        logger.info(f"라이브 스트림 연결 성공: 채널 {channel_id}")
        return cap

    def get_http_playback_url(self, channel_id: str, start: datetime, end: datetime) -> str:
        """HTTP 프로토콜을 위한 playback URL 생성"""
        # ISO 8601 UTC 문자열 생성
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 제조사별 URL 형식 (예시)
        params = {
            'starttime': start_iso,
            'endtime': end_iso,
            'channel': channel_id
        }
        
        # HTTP playback URL 구성
        playback_url = (
            f"http://{self.username}:{self.password}@{self.host}:{self.port}"
            f"{self.stream_path}/playback?{urlencode(params)}"
        )
        
        return playback_url

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

        # 프로토콜별 playback URL 구성
        if self.protocol.upper() == "RTSP":
            # ISO 8601 UTC 문자열 생성
            start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            playback_url = (
                f"rtsp://{self.username}:{self.password}"
                f"@{self.host}:{self.port}"
                f"/cam/playback?"
                f"channel={channel_id}&"
                f"starttime={start_iso}&"
                f"endtime={end_iso}&"
                f"subtype=0"
            )
        
        elif self.protocol.upper() == "HTTP":
            # HTTP playback URL
            playback_url = self.get_http_playback_url(channel_id, start, end)
            
            # HTTP API로 playback 세션 시작 (선택적)
            try:
                playback_data = {
                    "channel": channel_id,
                    "startTime": start.isoformat(),
                    "endTime": end.isoformat(),
                    "speed": 1.0
                }
                response = self._make_http_request("/playback/start", "POST", playback_data)
                session_id = response.json().get("sessionId")
                if session_id:
                    logger.info(f"HTTP playback 세션 시작: {session_id}")
                    
                    # 세션 ID를 URL에 추가
                    separator = "&" if "?" in playback_url else "?"
                    playback_url += f"{separator}session={session_id}"
                    
            except Exception as e:
                logger.warning(f"HTTP playback 세션 생성 실패, 직접 스트림 사용: {str(e)}")
        
        else:
            # ONVIF의 경우 라이브 스트림 URL 사용 (playback 구현은 복잡함)
            playback_url = channel.stream_url

        logger.info(f"Playback URL: {playback_url}")

        # OpenCV로 재생
        cap = cv2.VideoCapture()
        
        # HTTP의 경우 추가 설정
        if self.protocol.upper() == "HTTP":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        success = cap.open(playback_url, cv2.CAP_FFMPEG)
        if not success or not cap.isOpened():
            raise NVRConnectionError(f"Failed to open playback stream: {playback_url}")

        try:
            frame_count = 0
            consecutive_failures = 0
            max_failures = 10
            
            while consecutive_failures < max_failures:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning(f"연속 {max_failures}회 프레임 읽기 실패, 종료")
                        break
                    time.sleep(0.1)  # 짧은 대기
                    continue
                
                consecutive_failures = 0
                frame_count += 1
                yield frame
            
            logger.info(f"총 {frame_count}개 프레임 처리 완료")
            
        except Exception as e:
            logger.error(f"프레임 캡처 중 오류: {str(e)}")
            raise NVRError(f"프레임 캡처 실패: {str(e)}")
        finally:
            cap.release()
            
            # HTTP playback 세션 종료 (선택적)
            if self.protocol.upper() == "HTTP":
                try:
                    self._make_http_request("/playback/stop", "POST", {"channel": channel_id})
                except:
                    pass

    def capture_snapshot(self, channel_id: str, timestamp: datetime = None):
        """지정 시각의 스냅샷 캡처"""
        if self.protocol.upper() == "HTTP" and timestamp:
            # HTTP API를 통한 특정 시점 스냅샷
            try:
                snapshot_data = {
                    "channel": channel_id,
                    "timestamp": timestamp.isoformat() if timestamp else None
                }
                response = self._make_http_request("/snapshot", "POST", snapshot_data)
                
                if response.headers.get('content-type', '').startswith('image/'):
                    # 이미지 데이터를 numpy 배열로 변환
                    import numpy as np
                    nparr = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        logger.info(f"HTTP 스냅샷 캡처 성공: 채널 {channel_id}")
                        return frame
                    
            except Exception as e:
                logger.warning(f"HTTP 스냅샷 캡처 실패, 라이브 스트림 사용: {str(e)}")
        
        # 기본 방법: 라이브 스트림에서 캡처
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
        """장치 정보 조회"""
        if self.protocol.upper() == "ONVIF":
            onvif_camera = self._get_onvif_camera()
            return onvif_camera.get_device_information()
        
        elif self.protocol.upper() == "HTTP":
            try:
                response = self._make_http_request("/device/info")
                return response.json()
            except Exception as e:
                logger.error(f"HTTP 장치 정보 조회 실패: {str(e)}")
                return {"error": str(e)}
        
        else:
            raise NotImplementedError(f"장치 정보 조회는 {self.protocol} 프로토콜에서 지원되지 않습니다.")

    def control_ptz(self, channel_id: str, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0):
        """PTZ 제어"""
        if self.protocol.upper() == "ONVIF":
            channel = self._get_channel(channel_id)
            if not channel.profile_token:
                raise NVRError(f"채널 {channel_id}의 프로필 토큰이 없습니다.")
            
            onvif_camera = self._get_onvif_camera()
            onvif_camera.move_ptz(channel.profile_token, pan, tilt, zoom)
        
        elif self.protocol.upper() == "HTTP":
            ptz_data = {
                "channel": channel_id,
                "pan": pan,
                "tilt": tilt,
                "zoom": zoom
            }
            try:
                response = self._make_http_request("/ptz/control", "POST", ptz_data)
                logger.info(f"HTTP PTZ 제어 성공: {response.status_code}")
            except Exception as e:
                raise NVRError(f"HTTP PTZ 제어 실패: {str(e)}")
        
        else:
            raise NotImplementedError(f"PTZ 제어는 {self.protocol} 프로토콜에서 지원되지 않습니다.")
        
        logger.info(f"PTZ 제어 실행: 채널 {channel_id}, pan={pan}, tilt={tilt}, zoom={zoom}")

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            if self.protocol.upper() == "ONVIF":
                onvif_camera = self._get_onvif_camera()
                return onvif_camera.test_connection()
            elif self.protocol.upper() == "HTTP":
                response = self._make_http_request("/status")
                return response.status_code == 200
            else:
                # RTSP는 채널 목록 조회로 테스트
                channels = self.list_channels()
                return len(channels) > 0
        except:
            return False

    def get_recording_list(self, channel_id: str, start: datetime, end: datetime) -> List[Dict]:
        """녹화 파일 목록 조회 (HTTP 전용)"""
        if self.protocol.upper() != "HTTP":
            raise NotImplementedError("녹화 파일 목록 조회는 HTTP 프로토콜만 지원합니다.")
        
        try:
            params = {
                "channel": channel_id,
                "startTime": start.isoformat(),
                "endTime": end.isoformat()
            }
            response = self._make_http_request(f"/recordings?{urlencode(params)}")
            recordings = response.json()
            
            if isinstance(recordings, list):
                return recordings
            elif isinstance(recordings, dict) and "recordings" in recordings:
                return recordings["recordings"]
            else:
                return []
                
        except Exception as e:
            logger.error(f"녹화 파일 목록 조회 실패: {str(e)}")
            return []

    def __del__(self):
        """소멸자에서 HTTP 세션 정리"""
        if hasattr(self, '_http_session') and self._http_session:
            self._http_session.close()