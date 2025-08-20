from kafka import KafkaConsumer, TopicPartition
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import json
import time
import logging
from typing import Callable, Optional, Any, Dict, List
from threading import Event

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- dotenv loader (producers.py와 동일 철학) --------------------
def _load_env_best_effort():
    tried = set()
    for p in (
        os.getenv("DOTENV_PATH"),
        "/home/hiperwall/Ai_modules/Ai/env/aws.env",
        "env/aws.env",
        ".env",
        "../env/aws.env"
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
    try:
        load_dotenv(override=False)
    except Exception:
        pass


# -------------------- Kafka Murmur2 (Java 호환) --------------------
def _murmur2(data: bytes) -> int:
    """Kafka(Java) 호환 Murmur2 해시 (positive only)"""
    seed = 0x9747b28c
    m = 0x5bd1e995
    r = 24

    length = len(data)
    h = (seed ^ length) & 0xFFFFFFFF
    length4 = length & ~0x03

    for i in range(0, length4, 4):
        k = (data[i + 0] & 0xFF) | ((data[i + 1] & 0xFF) << 8) | ((data[i + 2] & 0xFF) << 16) | ((data[i + 3] & 0xFF) << 24)
        k = (k * m) & 0xFFFFFFFF
        k ^= (k >> r) & 0xFFFFFFFF
        k = (k * m) & 0xFFFFFFFF
        h = (h * m) & 0xFFFFFFFF
        h ^= k

    rem = length & 0x03
    if rem == 3:
        h ^= (data[length4 + 2] & 0xFF) << 16
        h ^= (data[length4 + 1] & 0xFF) << 8
        h ^= (data[length4 + 0] & 0xFF)
        h = (h * m) & 0xFFFFFFFF
    elif rem == 2:
        h ^= (data[length4 + 1] & 0xFF) << 8
        h ^= (data[length4 + 0] & 0xFF)
        h = (h * m) & 0xFFFFFFFF
    elif rem == 1:
        h ^= (data[length4 + 0] & 0xFF)
        h = (h * m) & 0xFFFFFFFF

    h ^= (h >> 13) & 0xFFFFFFFF
    h = (h * m) & 0xFFFFFFFF
    h ^= (h >> 15) & 0xFFFFFFFF
    # Kafka는 양수로 변환
    return h & 0x7fffffff


def _partition_for_key(consumer: KafkaConsumer, topic: str, key: str) -> int:
    """현재 클러스터 메타데이터 기준으로 camera_id가 갈 파티션 계산"""
    parts = consumer.partitions_for_topic(topic)
    if not parts:
        raise RuntimeError(f"Topic metadata not found: {topic}")
    num = len(parts)
    return _murmur2(key.encode("utf-8")) % num


# -------------------- FrameConsumer --------------------
class FrameConsumer:
    """
    - FRAME_TOPIC에서 해당 camera_id가 매핑되는 파티션만 assign
    - JPEG bytes → numpy frame 복원
    - handler(frame, headers) 콜백 호출 (예: DetectorAndTracker.detect_and_track)
    """
    def __init__(
        self,
        camera_id: str,
        handler: Callable[[np.ndarray, Dict[str, Any]], Any],
        *,
        auto_offset: str = "latest",
        poll_timeout_ms: int = 200,
    ):
        _load_env_best_effort()
        self.camera_id = str(camera_id)
        self.handler = handler

        self.broker = os.getenv("BROKER")
        self.topic = os.getenv("FRAME_TOPIC")

        if not self.broker or not self.topic:
            raise ValueError("BROKER 또는 FRAME_TOPIC 환경변수가 설정되지 않았습니다.")

        # value는 bytes 그대로, key도 bytes
        self.consumer = KafkaConsumer(
            bootstrap_servers=[self.broker],
            enable_auto_commit=False,
            auto_offset_reset=auto_offset,
        )
        self.stop_event = Event()

        # 파티션 계산 및 assign
        p = _partition_for_key(self.consumer, self.topic, self.camera_id)
        tp = TopicPartition(self.topic, p)
        self.consumer.assign([tp])

        # latest부터 받고 싶으면 seek_to_end, 과거 포함이면 auto_offset_reset='earliest' + seek 생략
        if auto_offset == "latest":
            try:
                self.consumer.seek_to_end(tp)
            except Exception:
                pass
        
        ''' for debugging'''
        pos = self.consumer.position(tp)
        logger.info(f"[FrameConsumer] assigned tp={tp} position={pos}")
        
        try:
            pos = self.consumer.position(tp)
            end = self.consumer.end_offsets([tp])[tp]
            logger.info(f"[FrameConsumer] assigned tp={tp} position={pos} end={end}")
        except Exception as e:
            logger.warning(f"[FrameConsumer] pos/end check failed: {e}")

        logger.info(f"[FrameConsumer] camera={self.camera_id} -> topic={self.topic} partition={p}")

        self.poll_timeout_ms = int(poll_timeout_ms)

    def run(self):
        try:
            while not self.stop_event.is_set():
                records = self.consumer.poll(timeout_ms=self.poll_timeout_ms)
                if not records:
                    continue

                if time.time() - last_log >= 1.0:
                    try:
                        tp = next(iter(self.consumer.assignment()))
                        pos = self.consumer.position(tp)
                        end = self.consumer.end_offsets([tp])[tp]
                        logger.info(f"[FrameConsumer] pos={pos} end={end}")
                    except Exception:
                        pass
                    last_log = time.time()

                for tp, msgs in records.items():
                    for msg in msgs:
                        # 키 필터 (같은 파티션에 다른 camera_id가 올 가능성 희박하지만 안전하게 필터링)
                        if msg.key and msg.key.decode("utf-8") != self.camera_id:
                            continue

                        # JPEG bytes → frame
                        buf = np.frombuffer(msg.value, dtype=np.uint8)
                        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if frame is None:
                            logger.warning("Failed to decode frame; skip")
                            continue

                        headers = {k: (v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else v)
                                   for k, v in (msg.headers or [])}

                        # 사용자 핸들러 호출
                        try:
                            self.handler(frame, headers)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

        except KeyboardInterrupt:
            logger.info("FrameConsumer interrupted")
        finally:
            try:
                self.consumer.close()
            except Exception:
                pass
            logger.info("FrameConsumer closed")

    def stop(self):
        self.stop_event.set()


# -------------------- ReIDResponseConsumer --------------------
class ReIDResponseConsumer:
    """
    - REID_RESPONSE_TOPIC에서 해당 camera_id가 매핑되는 파티션만 assign
    - value: JSON → dict
    - handler(response_dict) 콜백 호출 (예: DetectorAndTracker.on_reid_response)
    """
    def __init__(
        self,
        camera_id: str,
        handler: Callable[[Dict[str, Any]], Any],
        *,
        auto_offset: str = "latest",
        poll_timeout_ms: int = 200,
    ):
        _load_env_best_effort()
        self.camera_id = str(camera_id)
        self.handler = handler

        self.broker = os.getenv("BROKER")
        self.topic = os.getenv("REID_RESPONSE_TOPIC")

        if not self.broker or not self.topic:
            raise ValueError("BROKER 또는 REID_RESPONSE_TOPIC 환경변수가 설정되지 않았습니다.")

        self.consumer = KafkaConsumer(
            bootstrap_servers=[self.broker],
            enable_auto_commit=True,             # 응답은 재처리 부담이 적어서 auto commit 허용
            auto_offset_reset=auto_offset,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        self.stop_event = Event()

        p = _partition_for_key(self.consumer, self.topic, self.camera_id)
        tp = TopicPartition(self.topic, p)
        self.consumer.assign([tp])

        if auto_offset == "latest":
            try:
                self.consumer.seek_to_end(tp)
            except Exception:
                pass
        
       

        logger.info(f"[ReIDResponseConsumer] camera={self.camera_id} -> topic={self.topic} partition={p}")

        self.poll_timeout_ms = int(poll_timeout_ms)

    def run(self):
        try:
            while not self.stop_event.is_set():
                
                records = self.consumer.poll(timeout_ms=self.poll_timeout_ms)
                if not records:
                    continue

                for tp, msgs in records.items():
                    for msg in msgs:
                        # 같은 파티션에 다른 camera_id 응답이 섞일 수 있어 필터링
                        if msg.key and msg.key.decode("utf-8") != self.camera_id:
                            continue

                        data = msg.value
                        if not isinstance(data, dict):
                            logger.warning("Invalid response payload; skip")
                            continue

                        # 안전하게 camera_id 매칭 확인
                        if str(data.get("camera_id")) != self.camera_id:
                            continue

                        try:
                            self.handler(data)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

        except KeyboardInterrupt:
            logger.info("ReIDResponseConsumer interrupted")
        finally:
            try:
                self.consumer.close()
            except Exception:
                pass
            logger.info("ReIDResponseConsumer closed")

    def stop(self):
        self.stop_event.set()


# -------------------- Factories --------------------
def create_frame_consumer(camera_id: str, handler: Callable[[np.ndarray, Dict[str, Any]], Any]) -> FrameConsumer:
    return FrameConsumer(camera_id=camera_id, handler=handler)

def create_reid_response_consumer(camera_id: str, handler: Callable[[Dict[str, Any]], Any]) -> ReIDResponseConsumer:
    return ReIDResponseConsumer(camera_id=camera_id, handler=handler)