# nvr_util/exceptions.py

class NVRError(Exception):
    """Base exception for NVR errors."""

class NVRConnectionError(NVRError):
    """호스트 연결 실패 시."""

class NVRAuthError(NVRError):
    """인증(Authentication) 실패 시."""

class NVRChannelNotFoundError(NVRError):
    """존재하지 않는 채널 조회 시."""
