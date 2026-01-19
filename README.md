# TissueTrackr

A PythonFastAPI backend exposes a websocket endpoint for a incoming video stream and a separate websocket endpoint for multiple viewing clients, while a React/Vite frontend is used to view and annotate the stream. The backend uses OpenCV to build a large reference-point map that supports tracking objects even when they move off screen, along with optical-flow-based local anchors.

For further details & demo, see our [video](https://www.youtube.com/watch?v=XFmkCJpiIlU) here.

To run the project locally, see instructions below.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

## Stream emulator
Instead of a real webcam, use the following to emulate a video stream using an existing video file

```bash
cd utils
python input_stream.py --video path_to_video
```