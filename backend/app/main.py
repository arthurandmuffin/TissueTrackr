import asyncio
import base64
import json
import os
from typing import Dict

from fastapi import FastAPI, WebSocketDisconnect, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from .models import AnnotationsIn, FramePinRequest, TrackingConfigUpdate
from .services.anchor_manager import AnchorManager

app = FastAPI(title="TissueTrackr Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://10.122.163.177:5173",
        "http://10.122.163  .177:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize anchor manager
anchor_manager = AnchorManager(max_features=500, max_detection_dimension=960)

# Store active clients for video streaming
active_clients: Dict[object, Dict[str, bool]] = {}

def _build_anchor_points(landmarks):
    return {
        "count": len(landmarks),
        "coordinates": [(int(lm.x), int(lm.y)) for lm in landmarks],
        "sizes": [lm.size for lm in landmarks],
        "angles": [lm.angle for lm in landmarks],
    }


async def _broadcast_frame(frame, frame_state):
    """
    Broadcasts frame & relevant metadata to clients subscribed to websocket endpoint
    """
    if not active_clients:
        return

    include_frame = any(
        pref.get("include_frame") for pref in active_clients.values()
    )
    frame_payload = None
    if include_frame:
        output_frame = anchor_manager.render_frame(
            frame, frame_state.landmarks, frame_state.annotations, frame_state.frame_id
        )
        ok, buffer = cv2.imencode(".jpg", output_frame)
        if ok:
            frame_payload = base64.b64encode(buffer).decode("ascii")

    payload = {
        "success": True,
        "frame_id": frame_state.frame_id,
        "global_landmarks": jsonable_encoder(frame_state.landmarks),
        "annotations": jsonable_encoder(frame_state.annotations),
    }

    for websocket, pref in list(active_clients.items()):
        message = dict(payload)
        if pref.get("include_frame") and frame_payload:
            message["frame_jpeg"] = frame_payload
        try:
            await websocket.send_json(message)
        except Exception:
            active_clients.pop(websocket, None)

@app.websocket("/ws/incoming-stream")
async def websocket_incoming_stream(websocket: WebSocket):
    """
    WebSocket endpoint for receiving video data in binary frame form.
    Process frames in real time and detect anchor points.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_bytes()
            
            # Decode the frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue
            
            # Process the frame using anchor manager
            frame_state = anchor_manager.process_frame(frame)
            
            # Send back anchor points and anchor state
            anchor_points = _build_anchor_points(frame_state.landmarks)
            response = {
                "success": True,
                "frame_id": frame_state.frame_id,
                "anchor_points": anchor_points,
                "global_landmarks": jsonable_encoder(frame_state.landmarks),
                "annotations": jsonable_encoder(frame_state.annotations),
            }
            
            await websocket.send_json(response)
            await _broadcast_frame(frame, frame_state)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Incoming stream error: {str(e)}")
    finally:
        anchor_manager.reset_tracking_state()
        await websocket.close()

# Websocket to subscribe to stream
@app.websocket("/ws/outgoing-stream")
async def websocket_outgoing_stream(websocket: WebSocket):
    """
    WebSocket endpoint for clients to view video streams.
    """
    await websocket.accept()
    active_clients[websocket] = {"include_frame": True}
    
    try:
        while True:
            # Keep connection alive and receive any control messages
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and "include_frame" in payload:
                active_clients[websocket]["include_frame"] = bool(
                    payload["include_frame"]
                )
            
    except WebSocketDisconnect:
        active_clients.pop(websocket, None)
    except Exception as e:
        print(f"Outgoing stream error: {str(e)}")
        active_clients.pop(websocket, None)
    finally:
        await websocket.close()

# Create annotation
@app.post("/annotations")
async def receive_annotations(payload: AnnotationsIn):
    """
    Endpoint for receiving incoming annotations (timestamp, frame_id, annotation_detail)
    """
    try:
        created = anchor_manager.register_annotations(payload)
        return JSONResponse(
            {
                "success": True,
                "count": len(created),
                "annotations": jsonable_encoder(created),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error receiving annotations: {str(e)}")

@app.post("/frames/pin")
async def pin_frame(payload: FramePinRequest):
    """
    Cache frames from specified frame onwards, prevents droppage through sliding window cache.
    """
    try:
        anchor_manager.pin_frame(payload.frame_id)
        return JSONResponse({"success": True, "frame_id": payload.frame_id})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/frames/unpin")
async def unpin_frame(payload: FramePinRequest):
    """
    Removes cache drop protection from unpinned frame to next pinned frame / sliding window
    """
    anchor_manager.unpin_frame(payload.frame_id)
    return JSONResponse({"success": True, "frame_id": payload.frame_id})

@app.get("/tracking/config")
async def get_tracking_config():
    """
    Get current detection configs
    """
    return JSONResponse(jsonable_encoder(anchor_manager.get_tracking_config()))

@app.post("/tracking/config")
async def update_tracking_config(payload: TrackingConfigUpdate):
    """
    Update detection configs
    """
    try:
        updated = anchor_manager.update_tracking_config(payload)
        return JSONResponse(jsonable_encoder(updated))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/annotations")
async def clear_annotations():
    """
    Remove all stored annotations from the backend.
    """
    try:
        count = anchor_manager.clear_annotations()
        return JSONResponse({"success": True, "count": count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing annotations: {str(e)}")
