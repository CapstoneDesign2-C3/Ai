
class NVRError(Exception):
    """NVR 관련 기본 예외 클래스"""
    def __init__(self, message="NVR error occurred"):
        super().__init__(message)
        self.message = message

class NVRConnectionError(NVRError):
    """NVR 연결 관련 예외 클래스"""
    def __init__(self, message="NVR connection failed"):
        super().__init__(message)

class NVRAuthError(NVRError):
    """NVR 인증 관련 예외 클래스"""
    def __init__(self, message="NVR authentication failed"):
        super().__init__(message)

class NVRChannelNotFoundError(NVRError):
    """NVR 채널을 찾을 수 없을 때의 예외 클래스"""
    def __init__(self, message="NVR channel not found"):
        super().__init__(message)