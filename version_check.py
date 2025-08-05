#!/usr/bin/env python3
"""
완전한 CUDA & TensorRT 버전 확인 스크립트
사용법: python3 check_versions.py
"""

import subprocess
import sys
import os

def run_command(cmd):
    """명령어 실행 및 결과 반환"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None

def check_cuda_versions():
    """CUDA 버전들 확인"""
    print("🚀 CUDA 환경 정보")
    print("=" * 50)
    
    # NVCC 버전 (컴파일러)
    nvcc_version = run_command("nvcc --version")
    if nvcc_version:
        # nvcc 출력에서 버전 추출
        for line in nvcc_version.split('\n'):
            if 'release' in line:
                version = line.split('release ')[1].split(',')[0]
                print(f"✅ CUDA Compiler (nvcc): {version}")
                break
    else:
        print("❌ CUDA Compiler (nvcc): 설치되지 않음")
    
    # NVIDIA-SMI 버전 (드라이버)
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits")
    if nvidia_smi:
        parts = nvidia_smi.split(', ')
        if len(parts) >= 2:
            print(f"✅ NVIDIA Driver: {parts[0]}")
            print(f"✅ CUDA Runtime (nvidia-smi): {parts[1]}")
    else:
        print("❌ NVIDIA Driver: 설치되지 않음 또는 GPU 없음")
    
    # CUDA 설치 경로들 확인
    cuda_paths = run_command("ls /usr/local/ | grep cuda")
    if cuda_paths:
        print(f"📁 CUDA 설치 경로들:")
        for path in cuda_paths.split('\n'):
            if path:
                print(f"   /usr/local/{path}")
    
    # 현재 CUDA 심볼릭 링크
    cuda_link = run_command("ls -la /usr/local/cuda")
    if cuda_link and '->' in cuda_link:
        target = cuda_link.split('->')[-1].strip()
        print(f"🔗 현재 CUDA 링크: /usr/local/cuda -> {target}")

def check_tensorrt_versions():
    """TensorRT 버전들 확인"""
    print("\n⚡ TensorRT 환경 정보")
    print("=" * 50)
    
    # Python TensorRT 버전
    try:
        import tensorrt as trt
        print(f"✅ TensorRT Python: {trt.__version__}")
        
        # TensorRT 빌더 테스트
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            print("✅ TensorRT Builder: 정상 작동")
        except Exception as e:
            print(f"⚠️  TensorRT Builder: 오류 - {e}")
            
    except ImportError:
        print("❌ TensorRT Python: 설치되지 않음")
    
    # pip으로 설치된 TensorRT 패키지들
    tensorrt_packages = run_command("pip list | grep -i tensorrt")
    if tensorrt_packages:
        print("📦 설치된 TensorRT 패키지들:")
        for package in tensorrt_packages.split('\n'):
            if package:
                print(f"   {package}")
    
    # 시스템 TensorRT 라이브러리
    trt_libs = run_command("find /usr -name '*nvinfer*' -type f 2>/dev/null | head -3")
    if trt_libs:
        print("📚 TensorRT 라이브러리 파일들:")
        for lib in trt_libs.split('\n'):
            if lib:
                print(f"   {lib}")
    
    # ldconfig에서 TensorRT 확인
    trt_ldconfig = run_command("ldconfig -p | grep nvinfer")
    if trt_ldconfig:
        print("🔗 로드된 TensorRT 라이브러리들:")
        for lib in trt_ldconfig.split('\n')[:3]:  # 상위 3개만
            if lib:
                print(f"   {lib.strip()}")

def check_pytorch_cuda():
    """PyTorch CUDA 호환성 확인"""
    print("\n🔥 PyTorch CUDA 정보")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ PyTorch CUDA: {torch.version.cuda}")
        print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 메모리 정보
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"📊 GPU 메모리: {memory_allocated:.1f}MB 사용, {memory_cached:.1f}MB 예약")
            
            # CUDA 연산 테스트
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor.T)
                print("✅ CUDA 연산 테스트: 통과")
            except Exception as e:
                print(f"❌ CUDA 연산 테스트: 실패 - {e}")
        else:
            print("❌ CUDA를 사용할 수 없습니다")
            
    except ImportError:
        print("❌ PyTorch: 설치되지 않음")

def check_other_packages():
    """기타 중요 패키지들 확인"""
    print("\n📦 기타 중요 패키지들")
    print("=" * 50)
    
    packages_to_check = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'), 
        ('pycuda', 'pycuda.driver'),
        ('cupy', 'cupy')
    ]
    
    for package_name, import_name in packages_to_check:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: 설치되지 않음")

def check_environment_variables():
    """환경 변수 확인"""
    print("\n🌍 환경 변수")
    print("=" * 50)
    
    env_vars = [
        'CUDA_HOME',
        'CUDA_PATH', 
        'LD_LIBRARY_PATH',
        'PATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '')
        if var in ['LD_LIBRARY_PATH', 'PATH']:
            # CUDA 관련 경로만 필터링
            cuda_paths = [path for path in value.split(':') if 'cuda' in path.lower()]
            if cuda_paths:
                print(f"✅ {var} (CUDA 관련):")
                for path in cuda_paths[:3]:  # 최대 3개만
                    print(f"   {path}")
            else:
                print(f"⚠️  {var}: CUDA 경로 없음")
        else:
            if value:
                print(f"✅ {var}: {value}")
            else:
                print(f"⚠️  {var}: 설정되지 않음")

def main():
    """메인 실행 함수"""
    print("🔍 CUDA & TensorRT 환경 완전 분석")
    print("=" * 60)
    
    check_cuda_versions()
    check_tensorrt_versions()
    check_pytorch_cuda()
    check_other_packages()
    check_environment_variables()
    
    print(f"\n✅ 환경 분석 완료!")
    print("📝 이 정보를 바탕으로 호환 패키지를 선택하세요.")

if __name__ == "__main__":
    main()