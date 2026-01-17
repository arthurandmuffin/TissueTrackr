import asyncio
import cv2
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/incoming-stream"
VIDEO_PATH = "/Users/arthurhuang/Downloads/Dataset/Echo/echo1.mp4"
JPEG_QUALITY = 80
FPS_LIMIT = 20  # set None to send as fast as possible

async def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    delay = (1 / FPS_LIMIT) if FPS_LIMIT else 0

    async with websockets.connect(WS_URL, origin="http://127.0.0.1:8000", max_size=50 * 1024 * 1024) as ws:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Encode frame as JPEG bytes (what your server expects)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            await ws.send(buf.tobytes())
            msg = await ws.recv()  # server JSON response
            print(f"Frame {frame_idx}: {msg}")

            frame_idx += 1
            if delay:
                await asyncio.sleep(delay)

    cap.release()

asyncio.run(main())
