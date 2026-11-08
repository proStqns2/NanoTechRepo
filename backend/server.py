from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MAX_FRAME_BYTES = 1_500_000
LANDMARK_CONFIDENCE = 0.2
MODEL_COMPLEXITY = int(os.getenv("NANOTECH_MODEL_COMPLEXITY", "1"))
if MODEL_COMPLEXITY not in (0, 1, 2):
    raise ValueError("NANOTECH_MODEL_COMPLEXITY doit être 0, 1 ou 2.")

SEGMENTATION_VALUE = os.getenv("NANOTECH_ENABLE_SEGMENTATION", "0").strip().lower()
if SEGMENTATION_VALUE not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
    raise ValueError("NANOTECH_ENABLE_SEGMENTATION doit être une valeur booléenne.")
SEGMENTATION_ENABLED = SEGMENTATION_VALUE in {"1", "true", "yes", "on"}
logger = logging.getLogger("nanotech.pose")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _normalize(vector: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-7:
        return np.asarray(fallback, dtype=np.float64)
    return vector / length


def _quaternion(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        values = (
            (rotation[2, 1] - rotation[1, 2]) / root,
            (rotation[0, 2] - rotation[2, 0]) / root,
            (rotation[1, 0] - rotation[0, 1]) / root,
            root * 0.25,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        root = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
        values = (
            root * 0.25,
            (rotation[0, 1] + rotation[1, 0]) / root,
            (rotation[0, 2] + rotation[2, 0]) / root,
            (rotation[2, 1] - rotation[1, 2]) / root,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        root = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
        values = (
            (rotation[0, 1] + rotation[1, 0]) / root,
            root * 0.25,
            (rotation[1, 2] + rotation[2, 1]) / root,
            (rotation[0, 2] - rotation[2, 0]) / root,
        )
    else:
        root = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
        values = (
            (rotation[0, 2] + rotation[2, 0]) / root,
            (rotation[1, 2] + rotation[2, 1]) / root,
            root * 0.25,
            (rotation[1, 0] - rotation[0, 1]) / root,
        )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
    return [float(value) for value in quaternion]


def _transform(position: np.ndarray, rotation: np.ndarray, scale: tuple[float, float, float]) -> dict[str, Any]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = position
    return {
        "position": [float(value) for value in position],
        "quaternion": _quaternion(rotation),
        "scale": [float(value) for value in scale],
        "matrix": [float(value) for value in matrix.flatten(order="F")],
    }


def _segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    reference: np.ndarray,
) -> dict[str, Any]:
    direction = _normalize(end - start, (0.0, 1.0, 0.0))
    right = reference - direction * float(np.dot(reference, direction))
    right = _normalize(right, (1.0, 0.0, 0.0))
    forward = _normalize(np.cross(right, direction), (0.0, 0.0, 1.0))
    right = _normalize(np.cross(direction, forward), (1.0, 0.0, 0.0))
    rotation = np.column_stack((right, direction, forward))
    length = max(float(np.linalg.norm(end - start)), 0.025)
    return _transform((start + end) * 0.5, rotation, (radius, length, radius))


class PoseSession:
    """Runs one stateful MediaPipe tracker for a single browser connection."""

    def __init__(self) -> None:
        self.tracker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=SEGMENTATION_ENABLED,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.65,
        )
        self.previous: np.ndarray | None = None
        self.previous_at = 0.0
        self.frames = 0
        self.fps_started = time.perf_counter()
        self.fps = 0.0

    def close(self) -> None:
        self.tracker.close()

    def process(self, encoded: bytes) -> dict[str, Any]:
        if not encoded or len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("Frame vide ou trop volumineuse.")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("L’image reçue n’est pas un JPEG valide.")
        height, width = image.shape[:2]
        if width < 160 or height < 120:
            raise ValueError("La résolution webcam doit être d’au moins 160 × 120.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        started = time.perf_counter()
        result = self.tracker.process(rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.frames += 1
        fps_elapsed = time.perf_counter() - self.fps_started
        if fps_elapsed >= 1.0:
            self.fps = self.frames / fps_elapsed
            self.frames = 0
            self.fps_started = time.perf_counter()
        payload: dict[str, Any] = {
            "type": "pose",
            "timestamp": time.time(),
            "image": {"width": width, "height": height},
            "processingMs": round(elapsed_ms, 2),
            "fps": round(self.fps, 1),
            "detected": False,
            "confidence": 0.0,
            "scale": 1.0,
            "landmarks": [],
            "transforms": {},
            "segmentation": {"enabled": SEGMENTATION_ENABLED, "coverage": 0.0},
        }
        if not result.pose_landmarks:
            self.previous = None
            return payload
        landmarks = result.pose_landmarks.landmark
        points = np.asarray(
            [[1.0 - item.x, item.y, item.z, item.visibility] for item in landmarks],
            dtype=np.float64,
        )
        now = time.perf_counter()
        if self.previous is not None and self.previous.shape == points.shape:
            alpha = min(max((now - self.previous_at) / 0.075, 0.18), 0.82)
            visibility = points[:, 3:4]
            points[:, :3] = self.previous[:, :3] + (points[:, :3] - self.previous[:, :3]) * alpha
            points[:, 3:4] = visibility
        self.previous = points.copy()
        self.previous_at = now
        payload["landmarks"] = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "visibility": float(np.clip(point[3], 0.0, 1.0)),
            }
            for point in points
        ]
        pose_confidence = float(np.mean(points[[11, 12, 23, 24], 3]))
        payload["confidence"] = round(pose_confidence, 3)
        payload["detected"] = pose_confidence >= 0.35
        shoulder_width = float(np.linalg.norm(points[12, :2] - points[11, :2]))
        payload["scale"] = round(float(np.clip(0.25 / max(shoulder_width, 0.06), 0.55, 1.8)), 3)
        payload["transforms"] = self._transforms(points, width, height)
        if result.segmentation_mask is not None:
            payload["segmentation"]["coverage"] = round(
                float(np.mean(result.segmentation_mask > 0.5)), 3
            )
        return payload

    @staticmethod
    def _transforms(points: np.ndarray, width: int, height: int) -> dict[str, dict[str, Any]]:
        aspect = width / max(height, 1)

        def world(index: int) -> np.ndarray | None:
            if points[index, 3] < LANDMARK_CONFIDENCE:
                return None
            x, y, z = points[index, :3]
            return np.asarray(
                [(x - 0.5) * 2.0 * aspect, (0.5 - y) * 2.0, -z * 0.35],
                dtype=np.float64,
            )

        def average(*items: np.ndarray | None) -> np.ndarray | None:
            valid = [item for item in items if item is not None]
            return np.mean(valid, axis=0) if valid else None

        transforms: dict[str, dict[str, Any]] = {}
        ls, rs, le, re = (world(index) for index in (11, 12, 13, 14))
        lw, rw, lh, rh = (world(index) for index in (15, 16, 23, 24))
        lk, rk, la, ra = (world(index) for index in (25, 26, 27, 28))
        lt, rt = world(31), world(32)
        shoulders, hips = average(ls, rs), average(lh, rh)
        shoulder_span = float(np.linalg.norm(rs - ls)) if ls is not None and rs is not None else 0.28
        reference = np.asarray((0.0, 0.0, 1.0))
        up = _normalize(shoulders - hips, (0.0, 1.0, 0.0)) if shoulders is not None and hips is not None else np.asarray((0.0, 1.0, 0.0))
        right_axis = _normalize(rs - ls, (1.0, 0.0, 0.0)) if ls is not None and rs is not None else np.asarray((1.0, 0.0, 0.0))
        forward = _normalize(np.cross(right_axis, up), (0.0, 0.0, 1.0))
        right_axis = _normalize(np.cross(up, forward), (1.0, 0.0, 0.0))
        torso_rotation = np.column_stack((right_axis, up, forward))
        if shoulders is not None and hips is not None:
            transforms["torso"] = _transform(
                (shoulders + hips) * 0.5,
                torso_rotation,
                (shoulder_span, max(float(np.linalg.norm(shoulders - hips)), shoulder_span), shoulder_span * 0.38),
            )
        nose, left_ear, right_ear = world(0), world(7), world(8)
        head = average(nose, left_ear, right_ear)
        if head is not None:
            transforms["head"] = _transform(
                head + np.asarray((0.0, shoulder_span * 0.12, 0.0)),
                torso_rotation,
                (shoulder_span * 0.45, shoulder_span * 0.52, shoulder_span * 0.4),
            )
        for side, shoulder, elbow, wrist, hip, knee, ankle, toe in (
            ("left", ls, le, lw, lh, lk, la, lt),
            ("right", rs, re, rw, rh, rk, ra, rt),
        ):
            if shoulder is not None:
                transforms[f"shoulder_{side}"] = _transform(
                    shoulder, torso_rotation, (shoulder_span * 0.25,) * 3
                )
            if shoulder is not None and elbow is not None:
                transforms[f"upper_arm_{side}"] = _segment(shoulder, elbow, shoulder_span * 0.16, forward)
            if elbow is not None and wrist is not None:
                transforms[f"forearm_{side}"] = _segment(elbow, wrist, shoulder_span * 0.14, forward)
            if wrist is not None:
                transforms[f"hand_{side}"] = _transform(wrist, torso_rotation, (shoulder_span * 0.17,) * 3)
            if hip is not None and knee is not None:
                transforms[f"thigh_{side}"] = _segment(hip, knee, shoulder_span * 0.2, forward)
            if knee is not None and ankle is not None:
                transforms[f"shin_{side}"] = _segment(knee, ankle, shoulder_span * 0.15, forward)
            if ankle is not None:
                foot = toe if toe is not None else ankle + np.asarray((0.0, -shoulder_span * 0.3, 0.0))
                transforms[f"foot_{side}"] = _segment(ankle, foot, shoulder_span * 0.17, forward)
        if shoulders is not None and hips is not None:
            transforms["reactor"] = _transform(
                (shoulders + hips) * 0.5 + forward * shoulder_span * 0.23,
                torso_rotation,
                (shoulder_span * 0.19,) * 3,
            )
        return transforms


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not FRONTEND.joinpath("index.html").is_file():
        raise RuntimeError(f"Frontend absent : {FRONTEND}")
    yield


app = FastAPI(title="Nanotech Suit-Up Pose Server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "vision": "MediaPipe Pose"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.websocket("/ws/pose")
async def pose_socket(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in {"http://127.0.0.1:8000", "http://localhost:8000"}:
        await websocket.close(code=1008, reason="Origine non autorisée.")
        return
    await websocket.accept()
    session: PoseSession | None = None
    logger.info("Connexion de suivi ouverte depuis %s", websocket.client)
    try:
        session = PoseSession()
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            encoded = message.get("bytes")
            if encoded is None:
                await websocket.send_json({"type": "error", "message": "Le serveur attend des images JPEG binaires."})
                continue
            if len(encoded) > MAX_FRAME_BYTES:
                await websocket.send_json({"type": "error", "message": "Image trop volumineuse; réduisez la résolution webcam."})
                continue
            try:
                payload = await run_in_threadpool(session.process, encoded)
                await websocket.send_json(payload)
            except ValueError as error:
                await websocket.send_json({"type": "error", "message": str(error)})
            except Exception:
                logger.exception("Erreur de traitement MediaPipe.")
                await websocket.send_json({"type": "error", "message": "Le traitement de la pose a échoué."})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Connexion WebSocket de pose interrompue.")
    finally:
        if session is not None:
            await run_in_threadpool(session.close)
        logger.info("Connexion de suivi fermée.")


app.mount("/js", StaticFiles(directory=FRONTEND / "js"), name="frontend-js")


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Profil vision : complexité MediaPipe %s, segmentation %s.",
        MODEL_COMPLEXITY,
        "activée" if SEGMENTATION_ENABLED else "désactivée",
    )
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
