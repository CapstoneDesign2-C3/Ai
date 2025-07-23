#!/usr/bin/env python3
"""
NVR HTTP API 디버깅 스크립트
실제 NVR의 API 구조와 인증 방식을 확인합니다.
"""

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import cv2
from datetime import datetime, timedelta
import json
import time

# NVR 설정
NVR_HOST = "192.168.1.18"
NVR_PORT = 80
USERNAME = "admin"
PASSWORD = "hiperwall2018"

def test_authentication():
    """다양한 인증 방식 테스트"""
    print("=" * 50)
    print("인증 방식 테스트")
    print("=" * 50)
    
    # 테스트할 엔드포인트들 (UWA 기반 NVR용으로 수정)
    test_endpoints = [
        "/",
        "/uwa/",
        "/uwa/api/",
        "/uwa/api/system/info",
        "/uwa/api/device/info",
        "/uwa/api/channels",
        "/api/system/info",
        "/api/device/info",
        "/api/channels",
        "/api/v1/status",
        "/api/v1/system",
        "/api/v1/channels",
        "/web/",
        "/cgi-bin/hi3510/param.cgi?cmd=getserverinfo",
        "/cgi-bin/configManager.cgi?action=getConfig&name=Global",
        "/ISAPI/System/deviceInfo",
        "/ISAPI/System/status",
        "/device/info"
    ]
    
    auth_methods = [
        ("No Auth", None),
        ("Basic Auth", HTTPBasicAuth(USERNAME, PASSWORD)),
        ("Digest Auth", HTTPDigestAuth(USERNAME, PASSWORD))
    ]
    
    working_endpoints = []
    
    for auth_name, auth in auth_methods:
        print(f"\n--- {auth_name} 테스트 ---")
        session = requests.Session()
        session.auth = auth
        
        for endpoint in test_endpoints:
            url = f"http://{NVR_HOST}:{NVR_PORT}{endpoint}"
            try:
                response = session.get(url, timeout=5)
                status = response.status_code
                content_type = response.headers.get('content-type', 'unknown')
                content_length = len(response.content)
                
                print(f"  {endpoint:<50} | {status} | {content_type} | {content_length} bytes")
                
                if status == 200:
                    working_endpoints.append((auth_name, endpoint, url))
                    
                    # 응답 내용 일부 출력 (텍스트인 경우)
                    if 'text' in content_type or 'json' in content_type:
                        preview = response.text[:200].replace('\n', ' ')
                        print(f"    Preview: {preview}...")
                        
            except Exception as e:
                print(f"  {endpoint:<50} | ERROR: {str(e)}")
    
    print(f"\n작동하는 엔드포인트들:")
    for auth, endpoint, url in working_endpoints:
        print(f"  {auth}: {url}")
    
    return working_endpoints

def test_stream_urls():
    """다양한 스트림 URL 패턴 테스트"""
    print("\n" + "=" * 50)
    print("스트림 URL 테스트")
    print("=" * 50)
    
    channel = "1"
    
    # 다양한 스트림 URL 패턴 (UWA 기반 포함)
    stream_patterns = [
        # UWA 기반 패턴 (우선 테스트)
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/uwa/api/stream/{channel}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/uwa/api/channels/{channel}/stream",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/uwa/api/channels/{channel}/snapshot",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/api/stream/{channel}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/api/channels/{channel}/stream",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/api/channels/{channel}/snapshot",
        
        # Hikvision 패턴
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/Streaming/channels/{channel}01/picture",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/ISAPI/Streaming/channels/{channel}01/picture",
        
        # Dahua 패턴
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/cgi-bin/snapshot.cgi?channel={channel}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1",
        
        # 일반적인 패턴
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/mjpeg/{channel}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/video{channel}.mjpg",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/cam/realmonitor?channel={channel}&subtype=0",
        
        # RTSP over HTTP
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/cam/realmonitor?channel={channel}&subtype=0"
    ]
    
    working_streams = []
    
    for url in stream_patterns:
        print(f"\n테스트 중: {url}")
        
        # HTTP 요청으로 먼저 테스트
        try:
            response = requests.get(url, timeout=10, stream=True)
            print(f"  HTTP 응답: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            if response.status_code == 200:
                # OpenCV로 스트림 테스트
                try:
                    cap = cv2.VideoCapture()
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    
                    if cap.open(url, cv2.CAP_FFMPEG):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"  ✓ OpenCV 스트림 성공! 프레임 크기: {frame.shape}")
                            working_streams.append(url)
                        else:
                            print(f"  ✗ OpenCV 프레임 읽기 실패")
                    else:
                        print(f"  ✗ OpenCV 스트림 열기 실패")
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"  ✗ OpenCV 오류: {str(e)}")
            
        except Exception as e:
            print(f"  ✗ HTTP 요청 실패: {str(e)}")
    
    print(f"\n작동하는 스트림 URL들:")
    for url in working_streams:
        print(f"  {url}")
    
    return working_streams

def test_playback_urls():
    """재생(playback) URL 패턴 테스트"""
    print("\n" + "=" * 50)
    print("Playback URL 테스트")
    print("=" * 50)
    
    channel = "1"
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    
    # 다양한 시간 포맷
    start_iso = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_timestamp = str(int(start_time.timestamp()))
    end_timestamp = str(int(end_time.timestamp()))
    start_format = start_time.strftime("%Y%m%d_%H%M%S")
    end_format = end_time.strftime("%Y%m%d_%H%M%S")
    
    playback_patterns = [
        # UWA 기반 Playback 패턴 (우선 테스트)
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/uwa/api/playback/{channel}?startTime={start_iso}&endTime={end_iso}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/uwa/api/channels/{channel}/playback?startTime={start_iso}&endTime={end_iso}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/api/playback/{channel}?startTime={start_iso}&endTime={end_iso}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/api/channels/{channel}/playback?startTime={start_iso}&endTime={end_iso}",
        
        # Hikvision ISAPI
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/ISAPI/ContentMgmt/playback/channels/{channel}01?startTime={start_iso}&endTime={end_iso}",
        
        # Dahua CGI
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/cgi-bin/playback.cgi?action=download&channel={channel}&starttime={start_format}&endtime={end_format}",
        
        # RTSP over HTTP (일반적)
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/cam/playback?channel={channel}&starttime={start_iso}&endtime={end_iso}&subtype=0",
        
        # 기타 패턴들
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/playback?channel={channel}&starttime={start_iso}&endtime={end_iso}",
        f"http://{USERNAME}:{PASSWORD}@{NVR_HOST}:{NVR_PORT}/cgi-bin/recordfinder.cgi?action=find&channel={channel}&starttime={start_format}&endtime={end_format}"
    ]
    
    working_playback = []
    
    for url in playback_patterns:
        print(f"\n테스트 중: {url}")
        
        try:
            # HEAD 요청으로 URL 유효성 확인
            response = requests.head(url, timeout=10)
            print(f"  HTTP HEAD 응답: {response.status_code}")
            
            if response.status_code in [200, 206, 302]:
                print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
                working_playback.append(url)
                
                # 실제 데이터 가져오기 시도
                try:
                    response = requests.get(url, timeout=10, stream=True)
                    chunk = next(response.iter_content(chunk_size=1024), b'')
                    if chunk:
                        print(f"  ✓ 데이터 수신 성공 ({len(chunk)} bytes)")
                    else:
                        print(f"  ✗ 데이터 없음")
                except Exception as e:
                    print(f"  ✗ 데이터 수신 실패: {str(e)}")
            
        except Exception as e:
            print(f"  ✗ 요청 실패: {str(e)}")
    
    print(f"\n작동 가능한 Playback URL들:")
    for url in working_playback:
        print(f"  {url}")
    
    return working_playback

def test_rtsp_urls():
    """RTSP URL 패턴 테스트"""
    print("\n" + "=" * 50)
    print("RTSP URL 테스트")
    print("=" * 50)
    
    channel = "1"
    
    rtsp_patterns = [
        # 표준 RTSP 패턴들
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/cam/realmonitor?channel={channel}&subtype=0",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/cam/realmonitor?channel={channel}&subtype=1",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/stream{channel}",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/ch{channel}/main",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/ch{channel}/sub",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/h264/{channel}/main/av_stream",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/h264/{channel}/sub/av_stream",
        
        # Hikvision 패턴
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/Streaming/Channels/{channel}01",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/Streaming/Channels/{channel}02",
        
        # Dahua 패턴
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/live{channel}.sdp",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/ch{channel:02d}/0",
        f"rtsp://{USERNAME}:{PASSWORD}@{NVR_HOST}:554/ch{channel:02d}/1"
    ]
    
    working_rtsp = []
    
    for url in rtsp_patterns:
        print(f"\n테스트 중: {url}")
        
        try:
            # OpenCV로 RTSP 스트림 테스트
            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.open(url):
                # 연결 테스트
                start_time = time.time()
                ret, frame = cap.read()
                elapsed_time = time.time() - start_time
                
                if ret and frame is not None:
                    print(f"  ✓ RTSP 스트림 성공! 프레임 크기: {frame.shape}, 응답시간: {elapsed_time:.2f}초")
                    working_rtsp.append(url)
                    
                    # 추가 프레임 테스트 (안정성 확인)
                    frame_count = 0
                    for i in range(5):
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                    
                    print(f"    추가 프레임 테스트: {frame_count}/5 성공")
                    
                else:
                    print(f"  ✗ RTSP 프레임 읽기 실패")
            else:
                print(f"  ✗ RTSP 스트림 열기 실패")
            
            cap.release()
            
        except Exception as e:
            print(f"  ✗ RTSP 오류: {str(e)}")
    
    print(f"\n작동하는 RTSP URL들:")
    for url in working_rtsp:
        print(f"  {url}")
    
    return working_rtsp

def test_uwa_specific():
    """UWA 기반 NVR 전용 테스트"""
    print("\n" + "=" * 50)
    print("UWA 기반 NVR 전용 테스트")
    print("=" * 50)
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)
    
    # /stw-cgi/recording/search.cgi?Channel=1&StartTime=20250723T090000Z&EndTime=20250723T100000Z
    # UWA 관련 엔드포인트들
    uwa_endpoints = [
        "/uwa/",
        "/uwa/#/",
        "/uwa/#/live",
        "/uwa/#/search"
        "/uwa/#/login",
        "/uwa/#/logout", 
        "/uwa/#/system/info",
        "/uwa/#/system/status",
        "/uwa/#/channels",
        "/uwa/#/users",
        "/uwa/#/config",
        "/uwa/#/version",
        "/uwa/#/build"
    ]
    
    uwa_results = {}
    
    for endpoint in uwa_endpoints:
        url = f"http://{NVR_HOST}:{NVR_PORT}{endpoint}"
        print(f"\n테스트 중: {endpoint}")
        
        try:
            response = session.get(url, timeout=10)
            status = response.status_code
            content_type = response.headers.get('content-type', 'unknown')
            
            print(f"  응답: {status} | {content_type} | {len(response.content)} bytes")
            
            if status == 200:
                if 'json' in content_type:
                    try:
                        data = response.json()
                        print(f"  ✓ JSON 데이터: {list(data.keys()) if isinstance(data, dict) else 'Array'}")
                        uwa_results[endpoint] = data
                    except:
                        print(f"  ✗ JSON 파싱 실패")
                elif 'html' in content_type:
                    if 'uwa' in response.text.lower():
                        print(f"  ✓ UWA 인터페이스 확인됨")
                    uwa_results[endpoint] = "HTML_CONTENT"
                else:
                    preview = response.text[:150].replace('\n', ' ')
                    print(f"  Preview: {preview}...")
                    uwa_results[endpoint] = response.text
            
            elif status == 401:
                print(f"  ⚠️ 인증 필요")
            elif status == 403:
                print(f"  ⚠️ 접근 금지")
            elif status == 404:
                print(f"  ✗ 엔드포인트 없음")
            else:
                print(f"  ✗ 기타 오류")
                
        except Exception as e:
            print(f"  ✗ 요청 실패: {str(e)}")
    
    return uwa_results
    """NVR 디바이스 정보 탐지"""
    print("\n" + "=" * 50)
    print("디바이스 정보 탐지")
    print("=" * 50)
    
    # 디바이스 정보를 가져올 수 있는 엔드포인트들 (UWA 기반 포함)
    discovery_endpoints = [
        # UWA 기반 API (우선 시도)
        "/uwa/api/system/info",
        "/uwa/api/device/info",
        "/uwa/api/channels",
        "/uwa/api/system/status",
        "/uwa/api/config",
        "/api/system/info",
        "/api/device/info", 
        "/api/channels",
        "/api/v1/channels",
        "/api/system/status",
        
        # 전통적인 CGI 인터페이스
        "/cgi-bin/hi3510/param.cgi?cmd=getserverinfo",
        "/cgi-bin/configManager.cgi?action=getConfig&name=VideoIn",
        "/cgi-bin/configManager.cgi?action=getConfig&name=General",
        
        # ISAPI (Hikvision)
        "/ISAPI/System/deviceInfo",
        "/ISAPI/System/capabilities",
        "/ISAPI/Streaming/channels",
        
        # 기타
        "/device/channels"
    ]
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(USERNAME, PASSWORD)
    
    device_info = {}
    
    for endpoint in discovery_endpoints:
        url = f"http://{NVR_HOST}:{NVR_PORT}{endpoint}"
        print(f"\n테스트 중: {endpoint}")
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                
                if 'json' in content_type:
                    try:
                        data = response.json()
                        print(f"  ✓ JSON 응답 수신")
                        print(f"    키들: {list(data.keys()) if isinstance(data, dict) else 'List 형태'}")
                        device_info[endpoint] = data
                    except:
                        print(f"  ✗ JSON 파싱 실패")
                
                elif 'xml' in content_type:
                    print(f"  ✓ XML 응답 수신 ({len(response.text)} bytes)")
                    preview = response.text[:300].replace('\n', ' ')
                    print(f"    Preview: {preview}...")
                    device_info[endpoint] = response.text
                
                else:
                    print(f"  ✓ 텍스트 응답 수신 ({len(response.text)} bytes)")
                    preview = response.text[:200].replace('\n', ' ')
                    print(f"    Preview: {preview}...")
                    device_info[endpoint] = response.text
            
            else:
                print(f"  ✗ 응답 코드: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ 요청 실패: {str(e)}")
    
    return device_info

def generate_report(auth_results, stream_results, playback_results, rtsp_results, device_info, uwa_results=None):
    """테스트 결과 보고서 생성"""
    print("\n" + "=" * 80)
    print("NVR API 디버깅 결과 보고서")
    print("=" * 80)
    
    print(f"\n타겟 NVR: {NVR_HOST}:{NVR_PORT}")
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"브라우저 접속 URL: http://{NVR_HOST}/uwa/#/error")
    
    # UWA 기반 NVR 여부 확인
    is_uwa_based = uwa_results and len(uwa_results) > 0
    if is_uwa_based:
        print(f"🔍 NVR 타입: UWA 기반 웹 인터페이스")
    
    print(f"\n📊 결과 요약:")
    print(f"  - 작동하는 인증 엔드포인트: {len(auth_results)}개")
    print(f"  - 작동하는 스트림 URL: {len(stream_results)}개")
    print(f"  - 작동하는 Playback URL: {len(playback_results)}개")
    print(f"  - 작동하는 RTSP URL: {len(rtsp_results)}개")
    print(f"  - 수집된 디바이스 정보: {len(device_info)}개 엔드포인트")
    if is_uwa_based:
        print(f"  - UWA API 엔드포인트: {len(uwa_results)}개")
    
    if stream_results:
        print(f"\n🎥 권장 라이브 스트림 URL:")
        print(f"  {stream_results[0]}")
    
    if rtsp_results:
        print(f"\n📡 권장 RTSP URL:")
        print(f"  {rtsp_results[0]}")
    
    if playback_results:
        print(f"\n⏮️ Playback URL 예시:")
        print(f"  {playback_results[0]}")
    
    if is_uwa_based:
        print(f"\n🌐 UWA 기반 NVR 특별 고려사항:")
        print("  - 웹 기반 관리 인터페이스 사용")
        print("  - /uwa/api/ 경로의 REST API 사용 가능성")
        print("  - 세션 기반 인증이 필요할 수 있음")
    
    print(f"\n💡 다음 단계 권장사항:")
    
    if stream_results or rtsp_results:
        print("  1. 위의 작동하는 URL을 사용하여 실시간 스트리밍 구현")
        print("  2. OpenCV 또는 FFmpeg을 활용한 비디오 처리")
    
    if playback_results:
        print("  3. Playback API를 활용한 녹화 영상 다운로드/재생 구현")
    
    if device_info:
        print("  4. 디바이스 정보 API를 활용한 채널 정보 자동 탐지")
    
    if is_uwa_based:
        print("  5. UWA API 문서 확인 및 인증 방식 분석")
        print("  6. 웹 인터페이스의 네트워크 트래픽 분석 권장")
    
    print("  7. 에러 처리 및 재연결 로직 구현")
    print("  8. 인증 토큰 관리 (필요한 경우)")
    
    # 브라우저 개발자 도구 사용 가이드
    print(f"\n🔧 추가 분석 방법:")
    print("  1. 브라우저에서 http://192.168.1.18/uwa/ 접속")
    print("  2. 개발자 도구(F12) → Network 탭 열기")
    print("  3. 페이지 새로고침하여 API 호출 패턴 분석")
    print("  4. 로그인 시도하여 인증 방식 확인")

def main():
    """메인 실행 함수"""
    try:
        print("NVR HTTP API 디버깅 스크립트 시작")
        print(f"대상: {NVR_HOST}:{NVR_PORT}")
        print(f"계정: {USERNAME}")
        
        # 1. 인증 및 기본 엔드포인트 테스트
        auth_results = test_authentication()
        
        # 2. UWA 기반 NVR 전용 테스트
        uwa_results = test_uwa_specific()
        
        # 3. 라이브 스트림 URL 테스트
        stream_results = test_stream_urls()
        
        # 4. RTSP URL 테스트
        rtsp_results = test_rtsp_urls()
        
        # 5. Playback URL 테스트
        playback_results = test_playback_urls()
        
        # 6. 디바이스 정보 탐지
        device_info = test_device_discovery()
        
        # 7. 결과 보고서 생성
        generate_report(auth_results, stream_results, playback_results, rtsp_results, device_info, uwa_results)
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()