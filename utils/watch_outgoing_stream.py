import argparse
import asyncio
import base64
import json
import math
import random
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np
import websockets

DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/outgoing-stream"
DEFAULT_ORIGIN = "http://127.0.0.1:8000"
DEFAULT_ANNOTATIONS_URL = "http://127.0.0.1:8000/annotations"
MAX_MESSAGE_BYTES = 50 * 1024 * 1024
ANNOTATION_FRAME_MIN = 15
ANNOTATION_FRAME_MAX = 60
CIRCLE_RADIUS = 40
CIRCLE_POINTS = 16


def _decode_frame(frame_jpeg: str | None) -> np.ndarray | None:
    if not frame_jpeg:
        return None
    try:
        raw = base64.b64decode(frame_jpeg)
    except (ValueError, TypeError):
        return None
    data = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _circle_points(center_x: float, center_y: float, radius: float) -> list[dict]:
    points = []
    for idx in range(CIRCLE_POINTS):
        angle = 2 * math.pi * idx / CIRCLE_POINTS
        points.append(
            {
                "x": center_x + radius * math.cos(angle),
                "y": center_y + radius * math.sin(angle),
            }
        )
    return points


def _build_annotation_payload(
    frame_id: str | None, frame: np.ndarray
) -> dict | None:
    height, width = frame.shape[:2]
    max_radius = max(8, min(CIRCLE_RADIUS, width // 4, height // 4))
    if width <= max_radius * 2 or height <= max_radius * 2:
        return None
    center_x = random.randint(max_radius, width - max_radius - 1)
    center_y = random.randint(max_radius, height - max_radius - 1)

    annotation = {
        "geometry_type": "polygon",
        "points": _circle_points(center_x, center_y, max_radius),
        "metadata": {"label": "random_circle", "radius": max_radius},
        "local_hint": {"patch_radius": max_radius * 2},
    }
    return {
        "frame_id": frame_id,
        "annotations": [annotation],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _post_annotation(url: str, payload: dict) -> bool:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except Exception:
        return False
    return True


async def _keepalive(ws, interval: float) -> None:
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            await ws.send(json.dumps({"include_frame": True}))
        except Exception:
            return


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the /ws/outgoing-stream video feed from TissueTrackr."
    )
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket URL.")
    parser.add_argument("--origin", default=DEFAULT_ORIGIN, help="Origin header.")
    parser.add_argument(
        "--annotations-url",
        default=DEFAULT_ANNOTATIONS_URL,
        help="HTTP endpoint for posting annotations.",
    )
    parser.add_argument(
        "--keepalive",
        type=float,
        default=5.0,
        help="Seconds between keepalive control messages (0 to disable).",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable the random circle annotation.",
    )
    args = parser.parse_args()

    async with websockets.connect(
        args.url, origin=args.origin, max_size=MAX_MESSAGE_BYTES
    ) as ws:
        await ws.send(json.dumps({"include_frame": True}))
        keepalive_task = asyncio.create_task(_keepalive(ws, args.keepalive))
        window_title = "TissueTrackr Outgoing Stream"
        frame_index = 0
        annotation_frame = random.randint(ANNOTATION_FRAME_MIN, ANNOTATION_FRAME_MAX)
        annotation_attempted = False
        try:
            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    message = message.decode("utf-8", errors="ignore")
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue

                frame = _decode_frame(payload.get("frame_jpeg"))
                if frame is not None:
                    frame_index += 1
                    if (
                        not args.no_annotate
                        and not annotation_attempted
                        and frame_index >= annotation_frame
                    ):
                        annotation_attempted = True
                        annotation_payload = _build_annotation_payload(
                            payload.get("frame_id"), frame
                        )
                        if annotation_payload is not None:
                            sent = await asyncio.to_thread(
                                _post_annotation, args.annotations_url, annotation_payload
                            )
                            if sent:
                                print("Posted random circle annotation.")
                            else:
                                print("Failed to post annotation.")
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            keepalive_task.cancel()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
