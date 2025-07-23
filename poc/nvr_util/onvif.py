# onvif.py
"""
ONVIF Camera 연결 및 스트림 관리를 위한 모듈
onvif-zeep 또는 onvif2-zeep 패키지가 필요합니다.
"""

import logging
from typing import List, Optional, Dict, Any
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

try:
    from onvif2 import ONVIFCamera as BaseONVIFCamera
except ImportError:
    try:
        from onvif import ONVIFCamera as BaseONVIFCamera
    except ImportError:
        raise ImportError("onvif-zeep 또는 onvif2-zeep 패키지를 설치해주세요: pip install onvif2-zeep")

# 로깅 설정
logging.getLogger('onvif').setLevel(logging.WARNING)

class ONVIFCamera:
    """
    ONVIF 프로토콜을 사용하는 IP 카메라 연결 클래스
    """
    
    def __init__(self, host: str, port: int, username: str, password: str, wsdl_dir: Optional[str] = None):
        """
        ONVIF 카메라 초기화
        
        Args:
            host: 카메라 IP 주소
            port: ONVIF 포트 (보통 80, 8080, 또는 8899)
            username: 인증 사용자명
            password: 인증 비밀번호
            wsdl_dir: WSDL 파일 디렉토리 경로 (선택사항)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        
        try:
            # ONVIF 카메라 객체 생성
            if wsdl_dir:
                self.camera = BaseONVIFCamera(host, port, username, password, wsdl_dir)
            else:
                self.camera = BaseONVIFCamera(host, port, username, password)
            
            # 서비스 객체들 초기화
            self._media_service = None
            self._ptz_service = None
            self._device_service = None
            self._profiles = None
            
        except Exception as e:
            raise ConnectionError(f"ONVIF 카메라 연결 실패 ({host}:{port}): {str(e)}")
    
    def create_media_service(self):
        """Media 서비스 객체 생성 및 반환"""
        if self._media_service is None:
            try:
                self._media_service = self.camera.create_media_service()
            except Exception as e:
                raise RuntimeError(f"Media 서비스 생성 실패: {str(e)}")
        return self._media_service
    
    def create_ptz_service(self):
        """PTZ 서비스 객체 생성 및 반환"""
        if self._ptz_service is None:
            try:
                self._ptz_service = self.camera.create_ptz_service()
            except Exception as e:
                raise RuntimeError(f"PTZ 서비스 생성 실패: {str(e)}")
        return self._ptz_service
    
    def create_device_service(self):
        """Device 서비스 객체 생성 및 반환"""
        if self._device_service is None:
            try:
                self._device_service = self.camera.create_devicemgmt_service()
            except Exception as e:
                raise RuntimeError(f"Device 서비스 생성 실패: {str(e)}")
        return self._device_service
    
    def get_profiles(self) -> List[Any]:
        """카메라의 모든 미디어 프로필 조회"""
        if self._profiles is None:
            try:
                media_service = self.create_media_service()
                self._profiles = media_service.GetProfiles()
            except Exception as e:
                raise RuntimeError(f"프로필 조회 실패: {str(e)}")
        return self._profiles
    
    def get_stream_uri(self, profile_token: str, stream_type: str = 'RTP-Unicast', protocol: str = 'RTSP') -> str:
        """
        스트림 URI 조회
        
        Args:
            profile_token: 미디어 프로필 토큰
            stream_type: 스트림 타입 ('RTP-Unicast', 'RTP-Multicast')
            protocol: 프로토콜 ('RTSP', 'HTTP')
        
        Returns:
            스트림 URI 문자열
        """
        try:
            media_service = self.create_media_service()
            
            request = media_service.create_type('GetStreamUri')
            request.StreamSetup = {
                'Stream': stream_type,
                'Transport': {'Protocol': protocol}
            }
            request.ProfileToken = profile_token
            
            response = media_service.GetStreamUri(request)
            return response.Uri
            
        except Exception as e:
            raise RuntimeError(f"스트림 URI 조회 실패: {str(e)}")
    
    def get_device_information(self) -> Dict[str, str]:
        """장치 정보 조회"""
        try:
            device_service = self.create_device_service()
            info = device_service.GetDeviceInformation()
            
            return {
                'manufacturer': getattr(info, 'Manufacturer', 'Unknown'),
                'model': getattr(info, 'Model', 'Unknown'),
                'firmware_version': getattr(info, 'FirmwareVersion', 'Unknown'),
                'serial_number': getattr(info, 'SerialNumber', 'Unknown'),
                'hardware_id': getattr(info, 'HardwareId', 'Unknown')
            }
        except Exception as e:
            raise RuntimeError(f"장치 정보 조회 실패: {str(e)}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """장치 기능 조회"""
        try:
            device_service = self.create_device_service()
            capabilities = device_service.GetCapabilities()
            
            return {
                'media': getattr(capabilities, 'Media', None),
                'ptz': getattr(capabilities, 'PTZ', None),
                'device': getattr(capabilities, 'Device', None),
                'events': getattr(capabilities, 'Events', None)
            }
        except Exception as e:
            raise RuntimeError(f"장치 기능 조회 실패: {str(e)}")
    
    def move_ptz(self, profile_token: str, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0):
        """
        PTZ 제어 (상대적 이동)
        
        Args:
            profile_token: 미디어 프로필 토큰
            pan: 좌우 이동 (-1.0 ~ 1.0)
            tilt: 상하 이동 (-1.0 ~ 1.0)
            zoom: 줌 (-1.0 ~ 1.0)
        """
        try:
            ptz_service = self.create_ptz_service()
            
            request = ptz_service.create_type('RelativeMove')
            request.ProfileToken = profile_token
            request.Translation = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
            
            ptz_service.RelativeMove(request)
            
        except Exception as e:
            raise RuntimeError(f"PTZ 제어 실패: {str(e)}")
    
    def stop_ptz(self, profile_token: str):
        """PTZ 동작 중지"""
        try:
            ptz_service = self.create_ptz_service()
            
            request = ptz_service.create_type('Stop')
            request.ProfileToken = profile_token
            request.PanTilt = True
            request.Zoom = True
            
            ptz_service.Stop(request)
            
        except Exception as e:
            raise RuntimeError(f"PTZ 중지 실패: {str(e)}")
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            self.get_device_information()
            return True
        except:
            return False

# 편의를 위한 헬퍼 함수들
def discover_onvif_cameras(timeout: int = 5) -> List[Dict[str, str]]:
    """
    네트워크에서 ONVIF 카메라 자동 검색
    
    Args:
        timeout: 검색 타임아웃 (초)
    
    Returns:
        발견된 카메라 정보 리스트
    """
    try:
        from onvif2 import ONVIFService
        service = ONVIFService()
        cameras = service.discover(timeout)
        
        result = []
        for camera in cameras:
            result.append({
                'host': camera.getHost(),
                'port': camera.getPort(),
                'types': camera.getTypes()
            })
        
        return result
    except Exception as e:
        print(f"카메라 검색 실패: {str(e)}")
        return []

def test_onvif_connection(host: str, port: int, username: str, password: str) -> bool:
    """
    ONVIF 연결 테스트
    
    Args:
        host: 카메라 IP
        port: ONVIF 포트
        username: 사용자명
        password: 비밀번호
    
    Returns:
        연결 성공 여부
    """
    try:
        camera = ONVIFCamera(host, port, username, password)
        return camera.test_connection()
    except:
        return False