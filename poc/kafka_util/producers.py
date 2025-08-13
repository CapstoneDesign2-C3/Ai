from kafka import KafkaProducer
from dotenv import load_dotenv
import json
import cv2
import os
import base64
import numpy as np
import time
from typing import Optional, Dict, Any
import logging
from threading import Event

# -------------------- 로깅 --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 공통 유틸 --------------------
def _load_env_best_effort():
    """
    다양한 경로를 시도하여 env 로드.
    - 시스템마다 aws.env 경로가 달라 혼용됨: reid_service/nvr에서 사용된 경로 모두 시도.
    """
    tried = set()
    for p in (
        os.getenv("DOTENV_PATH"),
        "/home/hiperwall/Ai_modules/Ai/env/aws.env",  # nvr_client, reid main에서 사용
        "env/aws.env",                                 # reid_service 내부에서 사용
        ".env",
    ):
        if not p or p in tried:
            continue
        tried.add(p)
        if os.path.exists(p):
            try:
                load_dotenv(p, override=False)
                logger.info(f"dotenv loaded: {p}")
                return
            except Exception:
                pass
    # 마지막으로 기본 .env 시도(없는 경우 조용히 패스)
    try:
        load_dotenv(override=False)
    except Exception:
        pass


def _now_ms() -> int:
    return int(time.time() * 1000)


# -------------------- BaseProducer --------------------
class BaseProducer:
    """Kafka Producer 기본 클래스 - 카메라ID 기반 파티션(키 해시)"""

    def __init__(
        self,
        topic_env_key: str,
        *,
        acks: Optional[object] = 0,   # 프레임: 0(저지연), ReID 요청/응답: 'all'
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 10,
        max_request_size: int = 10 * 1024 * 1024,  # JPEG 페이로드 대비 상향
        key_serializer=None,
        value_serializer=None,
    ):
        _load_env_best_effort()

        self.broker = os.getenv("BROKER")
        self.topic = os.getenv(topic_env_key)
        if not self.broker or not self.topic:
            raise ValueError(f"BROKER 또는 {topic_env_key} 환경변수가 설정되지 않았습니다.")

        config = {
            "bootstrap_servers": [self.broker],  # 프로젝트 내 다른 모듈과 일치
            "acks": acks,
            "retries": retries,
            "batch_size": batch_size,
            "linger_ms": linger_ms,
            "max_request_size": max_request_size,
            # api_version는 하드코딩하지 않음(자동 프로빙)
            # "compression_type": None  # JPEG면 압축 비효율 → 기본값 유지
        }
        if key_serializer:
            config["key_serializer"] = key_serializer
        if value_serializer:
            config["value_serializer"] = value_serializer

        self.producer = KafkaProducer(**config)
        self._sent = 0
        self._flush_interval = 10  # 10개마다 flush
        self._closed = Event()

        logger.info(f"✅ {self.__class__.__name__} init - topic={self.topic}")

    def _maybe_flush(self):
        self._sent += 1
        if self._sent % self._flush_interval == 0:
            self.producer.flush()

    def _send(self, *, key: Optional[bytes], value: bytes, headers: Optional[list] = None, partition: Optional[int] = None) -> Dict[str, Any]:
        if self._closed.is_set():
            return {"status_code": 499, "error": "producer closed"}
        try:
            fut = self.producer.send(
                topic=self.topic,
                key=key,
                value=value,
                headers=headers or [],
                partition=partition,
            )
            # 성공/실패 콜백 로깅(파티션 확인)
            def on_success(md):
                logger.info(f"[{self.__class__.__name__}] -> {md.topic} p{md.partition} off{md.offset}")

            def on_error(ex):
                logger.error(f"{self.__class__.__name__} send failed: {ex}")

            fut.add_callback(on_success)
            fut.add_errback(on_error)

            self._maybe_flush()
            return {"status_code": 200, "error": None, "partition": partition}
        except Exception as e:
            logger.error(f"❌ send error: {e}")
            return {"status_code": 500, "error": str(e)}

    def close(self):
        if self._closed.is_set():
            return
        try:
            self._closed.set()
            self.producer.flush()
            self.producer.close()
            logger.info(f"✅ {self.__class__.__name__} closed")
        except Exception as e:
            logger.error(f"❌ close error: {e}")


# -------------------- FrameProducer --------------------
class FrameProducer(BaseProducer):
    """
    프레임 전송용 Producer
    - key = camera_id → 동일 카메라 스트림은 동일 파티션으로 해시됨
    - value = JPEG bytes
    - headers에 메타데이터(ts_ms, w, h, fmt) 추가
    """

    def __init__(self, camera_id: str):
        super().__init__(
            topic_env_key="FRAME_TOPIC",
            acks="all",              # 실시간 스트림: 유실 허용(저지연)
            retries=5,
            linger_ms=10,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            value_serializer=lambda v: v,  # bytes 그대로
        )
        self.camera_id = str(camera_id)

    def send_message(self, frame: np.ndarray, quality: int = 90, fmt: str = ".jpg") -> Dict[str, Any]:
        try:
            if fmt == ".jpg":
                params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            else:
                params = []
            ok, buf = cv2.imencode(fmt, frame, params)
            if not ok:
                raise ValueError("frame encode failed")

            h, w = frame.shape[:2]
            headers = [
                ("ts_ms", str(_now_ms()).encode()),
                ("camera_id", self.camera_id.encode()),
                ("w", str(w).encode()),
                ("h", str(h).encode()),
                ("fmt", fmt.encode()),
            ]
            return self._send(
                key=self.camera_id.encode("utf-8"),
                value=buf.tobytes(),
                headers=headers,
            )
        except Exception as e:
            logger.error(f"❌ frame send failed: {e}")
            return {"status_code": 500, "error": str(e)}


# -------------------- TrackResultProducer --------------------
class TrackResultProducer(BaseProducer):
    """
    추적 결과(크롭) → ReID 서비스 요청
    - key = camera_id
    - value = JSON (crop_jpg: base64)  ← ReID 서비스 계약을 정확히 맞춤
    """

    def __init__(self, camera_id: str):
        super().__init__(
            topic_env_key="REID_REQUEST_TOPIC",  # ReIDService가 읽는 키명
            acks="all",          # 유실 불가
            retries=10,
            linger_ms=20,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            value_serializer=lambda obj: json.dumps(obj).encode("utf-8") if isinstance(obj, dict) else obj,
        )
        self.camera_id = str(camera_id)

    def send_crop_from_base64(
        self,
        crop_base64: str,
        *,
        track_id: Optional[int] = None,
        bbox: Optional[list] = None,
        confidence: Optional[float] = None,
        class_name: Optional[str] = None,
        local_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            msg = {
                "camera_id": self.camera_id,
                "crop_jpg": crop_base64,       # ReID 서비스가 기대하는 필드명
                "timestamp": _now_ms(),
                "image_format": "jpeg",
                "encoding": "base64",
            }
            if track_id is not None:
                msg["track_id"] = track_id
                msg["local_id"] = local_id if local_id is not None else track_id
            if bbox is not None:
                msg["bbox"] = bbox
            if confidence is not None:
                msg["confidence"] = confidence
            if class_name is not None:
                msg["class_name"] = class_name

            headers = [
                ("ts_ms", str(msg["timestamp"]).encode()),
                ("camera_id", self.camera_id.encode()),
                ("enc", b"base64"),
                ("fmt", b"jpeg"),
            ]
            return self._send(
                key=self.camera_id.encode("utf-8"),
                value=msg,  # value_serializer가 JSON으로 직렬화
                headers=headers,
            )
        except Exception as e:
            logger.error(f"❌ base64 crop send failed: {e}")
            return {"status_code": 500, "error": str(e)}

    def send_message(
        self,
        crop: np.ndarray,
        *,
        track_id: Optional[int] = None,
        bbox: Optional[list] = None,
        confidence: Optional[float] = None,
        class_name: Optional[str] = None,
        encoding: str = "base64",
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        encoding='base64' 권장(서비스 포맷 부합). 'binary'도 지원.
        """
        try:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("crop encode failed")

            if encoding == "base64":
                crop_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return self.send_crop_from_base64(
                    crop_b64,
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_name=class_name,
                    local_id=track_id,
                )
            elif encoding == "binary":
                headers = [
                    ("ts_ms", str(_now_ms()).encode()),
                    ("camera_id", self.camera_id.encode()),
                    ("enc", b"binary"),
                    ("fmt", b"jpeg"),
                ]
                return self._send(
                    key=self.camera_id.encode("utf-8"),
                    value=buf.tobytes(),
                    headers=headers,
                )
            else:
                raise ValueError(f"unsupported encoding: {encoding}")
        except Exception as e:
            logger.error(f"❌ track result send failed: {e}")
            return {"status_code": 500, "error": str(e)}


# -------------------- ReIDResponseProducer --------------------
class ReIDResponseProducer(BaseProducer):
    """
    ReID 응답 → 카메라별 detectAndTrack/컨슈머에 전달
    - key = camera_id
    - value = JSON
    """
    def __init__(self):
        super().__init__(
            topic_env_key="REID_RESPONSE_TOPIC",  # 서비스 측도 동일 키로 맞추세요
            acks="all",
            retries=10,
            linger_ms=10,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            value_serializer=lambda obj: json.dumps(obj).encode("utf-8") if isinstance(obj, dict) else obj,
        )

    def send_response(
        self,
        camera_id: str,
        global_id: int,
        *,
        local_id: Optional[int] = None,
        track_id: Optional[int] = None,
        elapsed: Optional[float] = None,
        status: str = "success",
    ) -> Dict[str, Any]:
        try:
            msg = {
                "camera_id": camera_id,
                "global_id": int(global_id),
                "timestamp": _now_ms(),
                "status": status,
            }
            if local_id is not None:
                msg["local_id"] = int(local_id)
            if track_id is not None:
                msg["track_id"] = int(track_id)
            if elapsed is not None:
                msg["elapsed"] = float(elapsed)

            headers = [
                ("ts_ms", str(msg["timestamp"]).encode()),
                ("camera_id", str(camera_id).encode()),
            ]
            return self._send(
                key=str(camera_id).encode("utf-8"),
                value=msg,
                headers=headers,
            )
        except Exception as e:
            logger.error(f"❌ reid response send failed: {e}")
            return {"status_code": 500, "error": str(e)}


# -------------------- Factories --------------------
def create_frame_producer(camera_id: str) -> FrameProducer:
    return FrameProducer(camera_id)

def create_track_result_producer(camera_id: str) -> TrackResultProducer:
    return TrackResultProducer(camera_id)

def create_reid_response_producer() -> ReIDResponseProducer:
    return ReIDResponseProducer()
