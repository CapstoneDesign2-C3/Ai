#!/bin/bash
# create_init_files.sh - 누락된 __init__.py 파일들을 생성

echo "[*] Creating missing __init__.py files..."

# 메인 poc 패키지
touch poc/__init__.py

# 각 서브 패키지들
touch poc/db_util/__init__.py
touch poc/kafka_util/__init__.py
touch poc/nvr_util/__init__.py
touch poc/reid_module/__init__.py
touch poc/tracking_module/__init__.py
touch poc/yolo_engine/__init__.py

echo "[*] All __init__.py files created successfully!"

# 확인
echo "[*] Package structure:"
find poc -name "__init__.py" -type f