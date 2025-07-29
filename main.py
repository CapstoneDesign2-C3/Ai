# ./poc/main.py
from multiprocessing import Process
import os
import time

def start_nvr():
    """NVR 클라이언트 시작 (GPU 사용하지 않음)"""
    from poc.nvr_util.nvr_client import NVRClient
    print("[*] Starting NVR Client")
    client = NVRClient()
    
    processes = []
    for channel in client.NVRChannelList:
        def run_channel(ch):
            try:
                print(f"[*] NVR connected: {ch.camera_id}")
                ch.connect()
                ch.receive()
            except Exception as e:
                print(f"[NVR Error] Camera {ch.camera_id}: {e}")
        
        p = Process(target=run_channel, args=(channel,))
        p.start()
        processes.append(p)
    
    # 모든 채널 프로세스 대기
    for p in processes:
        p.join()

def start_kafka_worker():
    """DetectorTracker 시작 (GPU 0 사용)"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0 고정
    
    try:
        from poc.kafka_util.kafka_worker import KafkaWorker
        print("[*] Starting DetectorTracker (YOLO + DeepSORT) on GPU 0")
        
        # 엔진 파일 경로 확인
        engine_path = 'poc/yolo_engine/yolo11m_fp16.engine'
        if not os.path.exists(engine_path):
            print(f"[Error] Engine file not found: {engine_path}")
            print(f"[*] Looking for engine files in poc/yolo_engine/:")
            if os.path.exists('poc/yolo_engine/'):
                for f in os.listdir('poc/yolo_engine/'):
                    if f.endswith('.engine'):
                        print(f"  Found: {f}")
            return
        
        worker = KafkaWorker(engine_path=engine_path)
        worker.run()
    except ImportError as e:
        print(f"[Error] Import failed: {e}")
        print("[*] Make sure __init__.py files exist in all poc subdirectories")
    except Exception as e:
        print(f"[Error] KafkaWorker failed: {e}")
        import traceback
        traceback.print_exc()

def start_reid_service():
    """ReID 서비스 시작 (GPU 1 사용, 없으면 GPU 0 공유)"""
    import os
    # GPU가 2개 이상 있다면 GPU 1 사용, 없으면 GPU 0 공유
    try:
        import torch
        if torch.cuda.device_count() > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            print("[*] Starting ReID Service (OSNet + FAISS) on GPU 1")
        else:
            print("[*] Starting ReID Service (OSNet + FAISS) on GPU 0 (shared)")
    except:
        pass
    
    try:
        # ReID 서비스를 별도 파일로 실행
        os.system("python3 poc/reid_module/main.py")
    except Exception as e:
        print(f"[Error] ReID Service failed: {e}")

if __name__ == "__main__":
    print("[*] Starting Full Pipeline")
    print(f"[*] Available GPUs: {os.popen('nvidia-smi -L').read().strip()}")

    # 프로세스 시작 순서 조정
    processes = []
    
    # 1. NVR 클라이언트 시작 (가장 먼저)
    p1 = Process(target=start_nvr)
    p1.start()
    processes.append(p1)
    time.sleep(3)  # NVR이 안정화될 때까지 대기
    
    # 2. Kafka Worker 시작 (GPU 집약적)
    p2 = Process(target=start_kafka_worker)
    p2.start()
    processes.append(p2)
    time.sleep(5)  # TensorRT 초기화 완료까지 대기
    
    # 3. ReID 서비스 시작 (마지막)
    p3 = Process(target=start_reid_service)
    p3.start()
    processes.append(p3)
    
    # 모든 프로세스 완료 대기
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[*] Shutting down all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("[*] All processes terminated")