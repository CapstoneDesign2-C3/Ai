"""
Refactored Kafka consumers for camera-partitioned streams.
- Unifies common logic in a small base class (partition assignment, metadata wait, poll loop, logging, stop/close).
- Fixes a bug where `last_log` was referenced without `self.` in FrameConsumer.run.
- Adds rate-limited position/end logging and graceful shutdown.
- Keeps the public API and factory functions stable.

How it connects to DetectorAndTracker.on_reid_response:
- ReIDResponseConsumer filters by camera_id and delivers dict payloads to the provided handler.
- In your code, the handler is DetectorAndTracker.on_reid_response(data) which updates in-memory state and writes DB rows.
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import Callable, Any, Dict, Optional
from threading import Event

import cv2
import numpy as np
from dotenv import load_dotenv
from kafka import KafkaConsumer, TopicPartition

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- dotenv loader (shared with producers.py philosophy) --------------------
def _load_env_best_effort() -> None:
    tried = set()
    for p in (
        os.getenv("DOTENV_PATH"),
        "/home/hiperwall/Ai_modules/Ai/env/aws.env",
        "env/aws.env",
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
    try:
        load_dotenv(override=False)
    except Exception:
        pass


# -------------------- Kafka Murmur2 (Java-compatible) --------------------
def _murmur2(data: bytes) -> int:
    seed = 0x9747b28c
    m = 0x5bd1e995
    r = 24

    length = len(data)
    h = (seed ^ length) & 0xFFFFFFFF
    length4 = length & ~0x03

    for i in range(0, length4, 4):
        k = (
            (data[i + 0] & 0xFF)
            | ((data[i + 1] & 0xFF) << 8)
            | ((data[i + 2] & 0xFF) << 16)
            | ((data[i + 3] & 0xFF) << 24)
        )
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
    return h & 0x7FFFFFFF  # positive-only like Kafka


def _partition_for_key(consumer: KafkaConsumer, topic: str, key: str) -> int:
    parts = consumer.partitions_for_topic(topic)
    if not parts:
        raise RuntimeError(f"Topic metadata not found: {topic}")
    num = len(parts)
    return _murmur2(key.encode("utf-8")) % num


# -------------------- Common base: single-partition, camera-keyed consumer --------------------
class _PartitionedConsumerBase:
    def __init__(
        self,
        *,
        camera_id: str,
        topic_env_key: str,
        enable_auto_commit: bool,
        auto_offset: str = "latest",
        poll_timeout_ms: int = 200,
        value_deserializer: Optional[Callable[[bytes], Any]] = None,
    ) -> None:
        _load_env_best_effort()

        self.camera_id = str(camera_id)
        self.stop_event = Event()
        self.poll_timeout_ms = int(poll_timeout_ms)
        self.last_log = 0.0

        self.broker = os.getenv("BROKER")
        self.topic = os.getenv(topic_env_key)
        if not self.broker or not self.topic:
            raise ValueError(f"BROKER 또는 {topic_env_key} 환경변수가 설정되지 않았습니다.")

        conf: Dict[str, Any] = {
            "bootstrap_servers": [self.broker],
            "enable_auto_commit": bool(enable_auto_commit),
            "auto_offset_reset": str(auto_offset),
        }
        if value_deserializer is not None:
            conf["value_deserializer"] = value_deserializer

        self.consumer = KafkaConsumer(**conf)
        # Ensure metadata exists (fresh topics may take a moment)
        self._ensure_metadata()

        # Assign the camera-partition only
        p = _partition_for_key(self.consumer, self.topic, self.camera_id)
        self.tp = TopicPartition(self.topic, p)
        self.consumer.assign([self.tp])

        if auto_offset == "latest":
            try:
                self.consumer.seek_to_end(self.tp)
            except Exception:
                pass

        try:
            pos = self.consumer.position(self.tp)
            end = self.consumer.end_offsets([self.tp])[self.tp]
            logger.info(
                f"[{self.__class__.__name__}] camera={self.camera_id} -> {self.topic} p={p} pos={pos} end={end}"
            )
        except Exception as e:
            logger.info(
                f"[{self.__class__.__name__}] camera={self.camera_id} -> {self.topic} p={p} (pos/end check failed: {e})"
            )

    def _ensure_metadata(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                parts = self.consumer.partitions_for_topic(self.topic)
                if parts:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        # fall-through: let assignment raise if still unavailable

    def _log_position_rate_limited(self, interval_s: float = 1.0) -> None:
        now = time.time()
        if now - self.last_log < interval_s:
            return
        self.last_log = now
        try:
            pos = self.consumer.position(self.tp)
            end = self.consumer.end_offsets([self.tp])[self.tp]
            logger.info(f"[{self.__class__.__name__}] pos={pos} end={end}")
        except Exception:
            pass

    # Subclasses must implement this
    def _handle_record(self, msg) -> None:  # pragma: no cover (runtime path)
        raise NotImplementedError

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                records = self.consumer.poll(timeout_ms=self.poll_timeout_ms)
                if not records:
                    self._log_position_rate_limited()
                    continue

                self._log_position_rate_limited()

                for _tp, msgs in records.items():
                    for msg in msgs:
                        try:
                            # Filter by camera key if present
                            if msg.key and msg.key.decode("utf-8") != self.camera_id:
                                continue
                            self._handle_record(msg)
                        except Exception as e:
                            logger.error(f"{self.__class__.__name__} handler error: {e}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__} interrupted")
        finally:
            try:
                self.consumer.close()
            except Exception:
                pass
            logger.info(f"{self.__class__.__name__} closed")

    def stop(self) -> None:
        self.stop_event.set()


# -------------------- FrameConsumer --------------------
class FrameConsumer(_PartitionedConsumerBase):
    """
    - Reads JPEG bytes from FRAME_TOPIC camera-partition
    - Decodes to numpy frame and calls handler(frame, headers)
    """
    def __init__(
        self,
        camera_id: str,
        handler: Callable[[np.ndarray, Dict[str, Any]], Any],
        *,
        auto_offset: str = "latest",
        poll_timeout_ms: int = 200,
    ) -> None:
        self.handler = handler
        super().__init__(
            camera_id=camera_id,
            topic_env_key="FRAME_TOPIC",
            enable_auto_commit=False,
            auto_offset=auto_offset,
            poll_timeout_ms=poll_timeout_ms,
            value_deserializer=None,
        )

    def _handle_record(self, msg) -> None:  # pragma: no cover
        # JPEG bytes -> frame
        buf = np.frombuffer(msg.value, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to decode frame; skip")
            return

        headers = {
            k: (v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else v)
            for k, v in (msg.headers or [])
        }
        self.handler(frame, headers)


# -------------------- ReIDResponseConsumer --------------------
class ReIDResponseConsumer(_PartitionedConsumerBase):
    """
    - Reads dict payloads from REID_RESPONSE_TOPIC camera-partition
    - Calls handler(response_dict) (e.g., DetectorAndTracker.on_reid_response)
    """
    def __init__(
        self,
        camera_id: str,
        handler: Callable[[Dict[str, Any]], Any],
        *,
        auto_offset: str = "latest",
        poll_timeout_ms: int = 200,
    ) -> None:
        self.handler = handler
        super().__init__(
            camera_id=camera_id,
            topic_env_key="REID_RESPONSE_TOPIC",
            enable_auto_commit=True,  # responses are cheap to reprocess
            auto_offset=auto_offset,
            poll_timeout_ms=poll_timeout_ms,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )

    def _handle_record(self, msg) -> None:  # pragma: no cover
        data = msg.value
        if not isinstance(data, dict):
            logger.warning("Invalid response payload; skip")
            return

        # Ensure camera_id match inside payload as well
        if str(data.get("camera_id")) != self.camera_id:
            return

        self.handler(data)


# -------------------- Factories --------------------
def create_frame_consumer(camera_id: str, handler: Callable[[np.ndarray, Dict[str, Any]], Any]) -> FrameConsumer:
    return FrameConsumer(camera_id=camera_id, handler=handler)


def create_reid_response_consumer(camera_id: str, handler: Callable[[Dict[str, Any]], Any]) -> ReIDResponseConsumer:
    return ReIDResponseConsumer(camera_id=camera_id, handler=handler)
