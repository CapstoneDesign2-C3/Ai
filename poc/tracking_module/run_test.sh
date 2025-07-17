#!/bin/bash
# Usage: ./run_test.sh <engine_path> <video_path> [output_dir] [camera_id]
# Defaults:
#   engine_path: yolov11.engine
#   video_path: input.mp4
#   output_dir: output_crops
#   camera_id: CAM_TEST

ENGINE_PATH=${1:-yolov11m.engine}
VIDEO_PATH=${2:-sample.mp4}
OUTPUT_DIR=${3:-output_crops}
CAMERA_ID=${4:-CAM_TEST}

echo "Starting Detection-Tracking Test"
echo "Engine: ${ENGINE_PATH}"
echo "Video: ${VIDEO_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Camera ID: ${CAMERA_ID}"

# Create output directory if not exists
test -d "${OUTPUT_DIR}" || mkdir -p "${OUTPUT_DIR}"

# Run the Python test script
python test_detection_tracking.py \
    --engine "${ENGINE_PATH}" \
    --video "${VIDEO_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --camera_id "${CAMERA_ID}"

echo "Test completed. Crops saved under ${OUTPUT_DIR}/id_<track_id>/"
