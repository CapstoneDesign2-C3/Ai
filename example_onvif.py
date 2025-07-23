# example_onvif.py
import sys
import os
import cv2
from datetime import datetime

# poc 폴더를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc'))

from nvr_util.nvr_client import NVRClient
from nvr_util.exceptions import NVRConnectionError, NVRChannelNotFoundError

def main():
    try:
        # 1) ONVIF 클라이언트 초기화
        print("ONVIF 클라이언트 초기화 중...")
        client = NVRClient(name="onvif_test")
        
        # 2) 연결 테스트
        if client.test_connection():
            print("✅ ONVIF 연결 성공!")
        else:
            print("❌ ONVIF 연결 실패")
            return
        
        # 3) 장치 정보 조회 (ONVIF만)
        if client.protocol.upper() == "ONVIF":
            try:
                device_info = client.get_device_info()
                print(f"📱 장치 정보:")
                print(f"  제조사: {device_info['manufacturer']}")
                print(f"  모델: {device_info['model']}")
                print(f"  펌웨어: {device_info['firmware_version']}")
                print(f"  시리얼: {device_info['serial_number']}")
            except Exception as e:
                print(f"⚠️  장치 정보 조회 실패: {e}")
        
        # 4) 채널 목록 조회
        print("\n📺 채널 목록:")
        channels = client.list_channels()
        for ch in channels:
            print(f"  채널 {ch.id}: {ch.name}")
            print(f"    스트림 URL: {ch.stream_url}")
            if ch.profile_token:
                print(f"    프로필 토큰: {ch.profile_token}")
        
        # 5) 라이브 스트림 테스트
        if channels:
            first_channel = channels[0].id
            print(f"\n🔴 채널 {first_channel} 라이브 스트림 테스트...")
            
            try:
                cap = client.get_live_stream(channel_id=first_channel)
                
                print("라이브 스트림 연결됨. ESC 키를 눌러 종료...")
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("프레임 읽기 실패")
                        break
                    
                    frame_count += 1
                    
                    # 프레임 표시
                    cv2.imshow(f'Live Stream - Channel {first_channel}', frame)
                    
                    # ESC 키로 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    
                    # 10초 후 자동 종료
                    if frame_count > 300:  # 약 10초 (30fps 기준)
                        print("자동 종료...")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                print(f"총 {frame_count}개 프레임 처리됨")
                
            except Exception as e:
                print(f"❌ 라이브 스트림 실패: {e}")
        
        # 6) 스냅샷 캡처 테스트
        if channels:
            try:
                print(f"\n📸 채널 {first_channel} 스냅샷 캡처...")
                snapshot = client.capture_snapshot(first_channel)
                
                # 스냅샷 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_ch{first_channel}_{timestamp}.jpg"
                cv2.imwrite(filename, snapshot)
                print(f"스냅샷 저장됨: {filename}")
                
            except Exception as e:
                print(f"❌ 스냅샷 캡처 실패: {e}")
        
        # 7) PTZ 제어 테스트 (ONVIF만)
        if client.protocol.upper() == "ONVIF" and channels:
            try:
                print(f"\n🎮 PTZ 제어 테스트...")
                # 우측으로 약간 이동
                client.control_ptz(first_channel, pan=0.1, tilt=0.0, zoom=0.0)
                print("PTZ 제어 명령 전송됨 (우측 이동)")
                
                import time
                time.sleep(2)
                
                # 원래 위치로 복귀
                client.control_ptz(first_channel, pan=-0.1, tilt=0.0, zoom=0.0)
                print("PTZ 제어 명령 전송됨 (좌측 이동)")
                
            except Exception as e:
                print(f"⚠️  PTZ 제어 실패: {e}")
        
    except NVRConnectionError as e:
        print(f"❌ NVR 연결 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()