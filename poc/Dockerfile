# # syntax=docker/dockerfile:1.6

# ############################
# # Base (conda env 생성)
# ############################
# FROM mambaorg/micromamba:1.5.8 AS base

# # micromamba RUN/SHELL 활성화
# ARG MAMBA_DOCKERFILE_ACTIVATE=1

# # 작업 디렉터리
# WORKDIR /app/poc

# # ── 빌드 도구 (C 확장 빌드용: cython-bbox 등)
# USER root
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential pkg-config python3-dev \
#  && rm -rf /var/lib/apt/lists/*
# USER $MAMBA_USER

# # ── conda 환경 생성 (pycuda=conda, onnxruntime-gpu 등은 pip에서)
# #    ※ 루트 컨텍스트에서 -f poc/Dockerfile 로 빌드하므로 경로는 poc/ 로 시작
# COPY poc/environment.yml ./environment.yml
# RUN micromamba create -y -n app -f environment.yml --channel-priority strict && \
#     micromamba clean -ya

# # 이후 RUN/CMD는 conda env(app) 안에서 실행
# SHELL ["micromamba", "run", "-n", "app", "/bin/bash", "-lc"]

# # ── 공용 코드 복사 (tracker/reid 공통 유틸)
# COPY poc/db_util                ./db_util
# COPY poc/kafka_util             ./kafka_util
# COPY poc/nvr_util               ./nvr_util
# COPY poc/cp_that.py             ./cp_that.py
# COPY poc/eval.py                ./eval.py
# COPY poc/smoke_test_one_cam.py  ./smoke_test_one_cam.py
# COPY poc/test_engine.py         ./test_engine.py

# # ── 공통 환경 변수
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     TZ=Asia/Seoul \
#     PYTHONPATH=/app:/app/poc

# ############################
# # tracker 이미지
# # - tracking_module + yolo_engine 포함
# ############################
# FROM base AS tracker
# WORKDIR /app/poc

# COPY poc/tracking_module ./tracking_module
# COPY poc/yolo_engine     ./yolo_engine

# # 엔진 경로(컨테이너 내부)
# ENV ENGINE_PATH=/app/poc/yolo_engine/yolo11m_fp16.engine

# # 런타임도 conda env에서 실행
# ENTRYPOINT ["micromamba", "run", "-n", "app", "--no-capture-output"]
# CMD ["python", "smoke_test_one_cam.py"]

# ############################
# # reid 이미지
# # - reid_module만 포함
# ############################
# FROM base AS reid
# WORKDIR /app/poc

# COPY poc/reid_module ./reid_module

# ENTRYPOINT ["micromamba", "run", "-n", "app", "--no-capture-output"]
# CMD ["python", "-m", "reid_module.reid_service"]

# syntax=docker/dockerfile:1.6

# 1) micromamba 기반 (이미 사용 중) + 캐시 마운트
FROM mambaorg/micromamba:1.5.8 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# 기본 환경 변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # PyTorch CUDA 12.4 전용 인덱스 (torch==2.5.1+cu124, torchvision==0.20.1+cu124)
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# 시스템 패키지 (OpenCV 런타임 등)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app/poc

# 2) 의존성 레이어를 최대한 위로 올려 캐시 활용
#    environment.yml 먼저 복사
COPY environment.yml /tmp/environment.yml

# 3) micromamba 캐시 마운트로 conda 솔브 속도↑
#    channel priority를 strict로 고정해 솔브 변동성↓
RUN --mount=type=cache,target=/opt/conda/pkgs,id=conda-pkgs \
    micromamba create -y -n tracker -f /tmp/environment.yml --channel-priority strict && \
    micromamba clean -a -y

# 4) pip 캐시 마운트로 wheel 재사용
RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n tracker python -c "import sys; print(sys.version)" && \
    micromamba run -n tracker pip --version

# 5) 소스 복사 (의존성 이후 → 코드 변경 시 재빌드 최소화)
COPY . /app/poc

# 6) 런타임 환경 설정
ENV MAMBA_DEFAULT_ENV=tracker \
    PATH=/opt/conda/envs/tracker/bin:$PATH

# (선택) 헬스체크: 파이썬/토치/텐서RT가 살아있는지 빠르게 확인
RUN micromamba run -n tracker python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "is_available:", torch.cuda.is_available())
try:
    import tensorrt as trt
    print("TensorRT:", trt.__version__)
except Exception as e:
    print("TensorRT import failed:", e)
PY

# 7) 최종 이미지 크기 축소
RUN micromamba clean -a -y && \
    find /opt/conda/envs/tracker -name '__pycache__' -type d -prune -exec rm -rf {} +

# 기본 엔트리포인트는 네 프로젝트에 맞춰 필요 시 교체
CMD ["bash"]
