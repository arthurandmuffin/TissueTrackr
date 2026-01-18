import asyncio
import base64
import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, WebSocketDisconnect, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from .models import AnnotationsIn
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
    if not active_clients:
        return

    include_frame = any(
        pref.get("include_frame") for pref in active_clients.values()
    )
    frame_payload = None
    if include_frame:
        output_frame = anchor_manager.render_frame(
            frame, frame_state.landmarks, frame_state.annotations
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
    WebSocket endpoint for receiving incoming video streams.
    
    Processes frames in real-time and detects anchor points.
    
    Expected message format from client:
    - Binary frame data (raw JPEG/PNG bytes)
    
    Server response:
    - JSON with anchor points detected in each frame
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
        await websocket.close()

@app.websocket("/ws/outgoing-stream")
async def websocket_outgoing_stream(websocket: WebSocket):
    """
    WebSocket endpoint for sending processed video streams to clients.
    
    Clients connect to receive processed frames with annotations.
    
    Server sends:
    - Processed frame data with anchor points visualized
    - Metadata about detected features
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

@app.post("/annotations")
async def receive_annotations(payload: AnnotationsIn):
    """
    Endpoint for receiving incoming annotations.
    
    Accepts annotation data (e.g., user markings, corrections, metadata).
    
    Expected payload:
    - frame_id: Identifier for the frame being annotated
    - annotations: List of annotation objects with coordinates and metadata
    - timestamp: When the annotation was created
    
    Returns:
    - Confirmation of annotation receipt and storage
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
