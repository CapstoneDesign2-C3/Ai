#!/bin/bash
# setup_env.sh - 실행 전 환경 설정

# CUDA 메모리 관리
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# TensorRT 최적화
export TRT_LOGGER_VERBOSITY=3

# PyTorch CUDA 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 실행
echo "[*] Environment configured"
echo "[*] CUDA Devices: $(nvidia-smi -L | wc -l)"
echo "[*] Starting application..."

python main.py