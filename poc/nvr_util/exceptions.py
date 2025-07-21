# nvr_util/exceptions.py

class NVRError(Exception):
    pass

class NVRConnectionError(NVRError):
    def __init__(self):
        super.__init__('NVR 연결에 실패하였습니다.')

class NVRAuthError(NVRError):
    def __init__(self):
        super.__init__('인증 과정에서 오류가 발생하였습니다.')

class NVRChannelNotFoundError(NVRError):
    def __init__(self):
        super.__init__('NVR 채널을 찾을 수 없습니다.')
