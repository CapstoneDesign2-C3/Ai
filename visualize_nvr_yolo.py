import cv2
import numpy as np
import torch
import gc
import time
import sys
import os

# CUDA 관련 imports
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # 중요: CUDA 컨텍스트 자동 초기화
    PYCUDA_AVAILABLE = True
except ImportError:
    print("⚠️  PyCUDA not available, using PyTorch CUDA only")
    PYCUDA_AVAILABLE = False

# 프로젝트 경로 추가
sys.path.append('/home/hiperwall/Ai_modules/new/poc')

from poc.tracking_module.tracker import DetectorTracker, initialize_cuda_context
from poc.nvr_util.nvr_client import NVRClient
from poc.nvr_util.exceptions import NVRConnectionError, NVRRecieveError

def initialize_cuda():
    """CUDA 환경 초기화 및 메모리 정리"""
    print("🔧 Initializing CUDA environment...")
    
    try:
        # PyTorch CUDA 초기화 및 정리
        if torch.cuda.is_available():
            print(f"📱 CUDA devices available: {torch.cuda.device_count()}")
            print(f"📱 Current device: {torch.cuda.current_device()}")
            print(f"📱 Device name: {torch.cuda.get_device_name()}")
            
            # CUDA 캐시 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # CUDA 디바이스 설정
            torch.cuda.set_device(0)
            
            # CUDA 컨텍스트 워밍업
            warmup_tensor = torch.zeros(1).cuda()
            del warmup_tensor
            torch.cuda.synchronize()
            
            # 메모리 상태 출력
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"🔋 GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
            
        else:
            print("❌ CUDA not available!")
            return False
            
        # PyCUDA 초기화 (가능한 경우)
        if PYCUDA_AVAILABLE:
            try:
                cuda_context = initialize_cuda_context()
                if cuda_context:
                    print(f"🔧 PyCUDA context initialized successfully")
            except Exception as e:
                print(f"⚠️  PyCUDA initialization warning: {e}")
        
        print("✅ CUDA environment initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ CUDA initialization failed: {e}")
        return False

def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def select_camera_channel(nvr_client):
    """사용할 카메라 채널 선택"""
    if not nvr_client.NVRChannelList:
        raise Exception("No camera channels available")
    
    if len(nvr_client.NVRChannelList) == 1:
        return nvr_client.NVRChannelList[0]
    
    print("📹 Available cameras:")
    for i, channel in enumerate(nvr_client.NVRChannelList):
        print(f"  {i}: Camera {channel.camera_id} ({channel.camera_ip})")
    
    # 기본적으로 첫 번째 카메라 선택 (실제 환경에서는 사용자 입력 받을 수 있음)
    selected_index = 0
    return nvr_client.NVRChannelList[selected_index]
    """환경 변수 설정"""
    print("🔧 Setting up environment variables...")
    
    # CUDA 환경 변수 설정
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # TensorRT 로깅 레벨 설정 (선택사항)
    os.environ['TRT_LOGGER_VERBOSITY'] = 'WARNING'
    
    print("✅ Environment variables set")

def main():
    print("🚀 Starting NVR YOLO Visualization with GPU optimization...")
    
    # 환경 설정
    
    # CUDA 초기화
    if not initialize_cuda():
        print("❌ Failed to initialize CUDA. Exiting...")
        return
    
    try:
        # DetectorTracker 초기화 전 메모리 정리
        cleanup_gpu_memory()
        print("🚀 Initializing DetectorTracker...")
        
        # DetectorTracker 초기화
        engine_path = "yolo11m_fp16.engine"
        detector_tracker = DetectorTracker(engine_path=engine_path)
        
        print("📊 Engine Info:", detector_tracker.get_engine_info())
        
        # NVR 클라이언트 초기화
        print("📡 Connecting to NVR...")
        nvr_client = NVRClient()
        
        # 카메라 채널 선택
        channel = select_camera_channel(nvr_client)
        print(f"📹 Selected camera: {channel.camera_id} ({channel.camera_ip})")
        
        # NVR 채널 연결
        try:
            channel.connect()
            print(f"✅ Connected to camera: {channel.camera_id}")
            print(f"📊 Camera info - Resolution: {channel.width}x{channel.height}, FPS: {channel.fps:.1f}")
        except NVRConnectionError as e:
            raise Exception(f"NVR connection failed: {e}")
        except Exception as e:
            raise Exception(f"Failed to connect to camera: {e}")
        
        print("🎥 Starting video processing... Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        connection_retry_count = 0
        max_retries = 5
        
        try:
            while True:
                # 프레임 가져오기 (NVR 방식)
                try:
                    ret, frame = channel.cap.read()
                    if not ret or frame is None:
                        print("⚠️  No frame received, attempting reconnection...")
                        
                        # 재연결 시도
                        if connection_retry_count < max_retries:
                            try:
                                channel.disconnect()
                                time.sleep(1)
                                channel.connect()
                                connection_retry_count += 1
                                print(f"🔄 Reconnection attempt {connection_retry_count}/{max_retries}")
                                continue
                            except Exception as reconnect_error:
                                print(f"❌ Reconnection failed: {reconnect_error}")
                                connection_retry_count += 1
                                time.sleep(2)
                                if connection_retry_count >= max_retries:
                                    raise Exception("Max reconnection attempts reached")
                                continue
                        else:
                            raise Exception("Max reconnection attempts reached")
                            
                    # 연결 성공 시 재시도 카운터 리셋
                    connection_retry_count = 0
                    
                except NVRRecieveError as nvr_error:
                    print(f"⚠️  NVR receive error: {nvr_error}")
                    time.sleep(0.1)
                    continue
                except Exception as frame_error:
                    print(f"⚠️  Frame read error: {frame_error}")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # 주기적으로 GPU 메모리 정리 (100프레임마다)
                if frame_count % 100 == 0:
                    cleanup_gpu_memory()
                    
                    # 메모리 사용량 출력
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**2
                        memory_reserved = torch.cuda.memory_reserved() / 1024**2
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        print(f"📊 Frame: {frame_count}, FPS: {fps:.1f}, "
                              f"GPU Memory: {memory_allocated:.1f}MB/{memory_reserved:.1f}MB")
                
                try:
                    # YOLO 추론 실행
                    results = detector_tracker.detect_and_track(frame)
                    
                    # 결과 시각화
                    annotated_frame = detector_tracker.visualize_results(frame, results)
                    
                    # 검출된 객체 수 표시 (주기적으로)
                    if frame_count % 100 == 0 and results:
                        print(f"🎯 Detected {len(results)} person(s) at frame {frame_count}")
                    
                    # 프레임 표시
                    cv2.imshow('NVR YOLO Detection', annotated_frame)
                    
                except Exception as inference_error:
                    print(f"⚠️  Inference error at frame {frame_count}: {inference_error}")
                    print(f"🔍 Error type: {type(inference_error).__name__}")
                    
                    # 추론 실패 시 원본 프레임 표시
                    cv2.imshow('NVR YOLO Detection', frame)
                    
                    # GPU 메모리 정리 시도
                    cleanup_gpu_memory()
                    time.sleep(0.1)  # 잠시 대기
                
                # 키 입력 확인
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quit signal received")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("🔧 Attempting GPU memory cleanup...")
        cleanup_gpu_memory()
        
    finally:
        print("🧹 Cleaning up resources...")
        
        # NVR 연결 해제
        try:
            if 'channel' in locals():
                channel.disconnect()
                print("📡 NVR channel disconnected")
        except Exception as e:
            print(f"⚠️  NVR disconnect warning: {e}")
        
        # 최종 정리
        cleanup_gpu_memory()
        
        # OpenCV 윈도우 정리
        cv2.destroyAllWindows()
        
        # 추가 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Cleanup completed")
        
        # 최종 메모리 상태 출력
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"🔋 Final GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")

if __name__ == "__main__":
    main()