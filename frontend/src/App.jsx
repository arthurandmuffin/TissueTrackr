import React, { useEffect, useRef, useState } from "react";

const WS_BASE = import.meta.env.VITE_WS_BASE || "ws://localhost:8000";
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [frameSrc, setFrameSrc] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [frameId, setFrameId] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [color, setColor] = useState("#ff5252");
  const [mode, setMode] = useState("draw");
  const [eraserSize, setEraserSize] = useState(18);
  const [strokes, setStrokes] = useState([]);
  const [status, setStatus] = useState("idle");
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasPending, setHasPending] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showPlaybackHint, setShowPlaybackHint] = useState(false);
  const [lastPlaybackAction, setLastPlaybackAction] = useState("pause");

  const frameRef = useRef(null);
  const overlayRef = useRef(null);
  const outgoingRef = useRef(null);
  const drawingRef = useRef(false);
  const currentStrokeRef = useRef([]);
  const currentStrokeColorRef = useRef("#ff5252");
  const hintTimerRef = useRef(null);
  const isPlayingRef = useRef(false);

  useEffect(() => {
    resizeOverlay();
    drawOverlay();
  }, [landmarks, strokes, color, frameSrc]);

  useEffect(() => {
    const handleResize = () => {
      resizeOverlay();
      drawOverlay();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    return () => {
      disconnectSockets();
    };
  }, []);

  const connectSockets = () => {
    disconnectSockets();

    const outgoing = new WebSocket(`${WS_BASE}/ws/outgoing-stream`);
    setStatus("connecting");
    outgoing.onopen = () => {
      setStatus("connected");
      outgoing.send(JSON.stringify({ include_frame: true }));
      setIsPlaying(true);
      isPlayingRef.current = true;
      setStreaming(true);
    };
    outgoing.onclose = () => {
      setStatus("closed");
      setStreaming(false);
    };
    outgoing.onerror = () => setStatus("error");
    outgoing.onmessage = handleOutgoingMessage;

    outgoingRef.current = outgoing;
  };

  const disconnectSockets = () => {
    if (outgoingRef.current) outgoingRef.current.close();
    outgoingRef.current = null;
    setStreaming(false);
    setStatus("idle");
    setFrameId(null);
    setLandmarks([]);
    setFrameSrc("");
    setStrokes([]);
    setHasPending(false);
    setIsPlaying(false);
    isPlayingRef.current = false;
    drawingRef.current = false;
    currentStrokeRef.current = [];
    currentStrokeColorRef.current = color;
  };

  const handleOutgoingMessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (!isPlayingRef.current) return;
      if (data.frame_jpeg) {
        setFrameSrc(`data:image/jpeg;base64,${data.frame_jpeg}`);
      }
      setFrameId(data.frame_id || null);
      setLandmarks(data.global_landmarks || []);
      drawOverlay();
    } catch (err) {
      console.error("Failed to parse message", err);
    }
  };

  const resizeOverlay = () => {
    const frame = frameRef.current;
    const overlay = overlayRef.current;
    if (!frame || !overlay) return;
    const rect = frame.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
  };

  const getScale = () => {
    const frame = frameRef.current;
    const overlay = overlayRef.current;
    if (!frame || !overlay || !frame.naturalWidth || !frame.naturalHeight) {
      return { scaleX: 1, scaleY: 1 };
    }
    return {
      scaleX: overlay.width / frame.naturalWidth,
      scaleY: overlay.height / frame.naturalHeight,
    };
  };

  const getImageSpacePoint = (clientX, clientY) => {
    const overlay = overlayRef.current;
    const frame = frameRef.current;
    if (!overlay || !frame) return null;
    const rect = overlay.getBoundingClientRect();
    const { scaleX, scaleY } = getScale();
    if (!scaleX || !scaleY) return null;
    return {
      x: (clientX - rect.left) / scaleX,
      y: (clientY - rect.top) / scaleY,
    };
  };

  const drawOverlay = () => {
    const overlay = overlayRef.current;
    const ctx = overlay?.getContext("2d");
    if (!overlay || !ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const { scaleX, scaleY } = getScale();

    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const allStrokes = [
      ...strokes,
      currentStrokeRef.current.length
        ? { points: currentStrokeRef.current, color: currentStrokeColorRef.current }
        : null,
    ].filter(Boolean);
    allStrokes.forEach((stroke) => {
      if (!stroke.points.length) return;
      ctx.strokeStyle = stroke.color || color;
      ctx.beginPath();
      stroke.points.forEach((pt, idx) => {
        const x = pt.x * scaleX;
        const y = pt.y * scaleY;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

  };

  const handleOverlayClick = (e) => {
    e.preventDefault();
    if (mode === "still") {
      togglePlay();
    }
  };

  const eraserCursor = () => {
    const size = Math.max(24, Math.min(80, Math.round(eraserSize * 2)));
    const stroke = Math.max(2, Math.round(size / 14));
    const center = size / 2;
    const radius = Math.max(4, center - stroke);
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}' viewBox='0 0 ${size} ${size}'><circle cx='${center}' cy='${center}' r='${radius}' fill='none' stroke='#ffffff' stroke-width='${stroke}' /></svg>`;
    let encoded = "";
    try {
      encoded = `data:image/svg+xml;base64,${btoa(svg)}`;
    } catch {
      encoded = `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
    }
    const url = `url("${encoded}") ${Math.round(center)} ${Math.round(center)}, crosshair`;
    return url;
  };

  const handlePointerDown = (e) => {
    if (e.button !== 0 && e.pointerType !== "touch") return;
    if (mode === "still") return;
    if (isPlayingRef.current) {
      togglePlay();
    }
    const point = getImageSpacePoint(e.clientX, e.clientY);
    if (!point) return;
    if (mode === "draw") {
      drawingRef.current = true;
      currentStrokeRef.current = [point];
      currentStrokeColorRef.current = color;
      setHasPending(true);
      drawOverlay();
      return;
    }
    if (mode === "erase") {
      drawingRef.current = true;
      eraseAtPoint(point);
      drawOverlay();
    }
  };

  const handlePointerMove = (e) => {
    if (mode === "still") return;
    if (!drawingRef.current) return;
    const point = getImageSpacePoint(e.clientX, e.clientY);
    if (!point) return;
    if (mode === "draw") {
      currentStrokeRef.current.push(point);
      drawOverlay();
      return;
    }
    if (mode === "erase") {
      eraseAtPoint(point);
      drawOverlay();
    }
  };

  const handlePointerUp = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    if (mode === "draw" && currentStrokeRef.current.length) {
      const stroke = {
        points: currentStrokeRef.current,
        color: currentStrokeColorRef.current,
      };
      setStrokes((prev) => [...prev, stroke]);
      setHasPending(true);
    }
    currentStrokeRef.current = [];
    drawOverlay();
  };

  const handlePointerLeave = () => {
    drawingRef.current = false;
    drawOverlay();
  };

  const togglePlay = () => {
    if (!streaming) return;
    if (hasPending) {
      alert("Send or cancel the current annotation before playing.");
      return;
    }
    const next = !isPlayingRef.current;
    isPlayingRef.current = next;
    setIsPlaying(next);
    setLastPlaybackAction(next ? "play" : "pause");
    showPlaybackOverlay();
  };

  const showPlaybackOverlay = () => {
    setShowPlaybackHint(true);
    if (hintTimerRef.current) clearTimeout(hintTimerRef.current);
    hintTimerRef.current = setTimeout(() => {
      setShowPlaybackHint(false);
    }, 1200);
  };

  const cancelAnnotation = () => {
    setStrokes([]);
    currentStrokeRef.current = [];
    drawingRef.current = false;
    setHasPending(false);
  };

  const eraseAll = () => {
    setStrokes([]);
    currentStrokeRef.current = [];
    drawingRef.current = false;
    setHasPending(false);
  };

  const eraseAtPoint = (point) => {
    const radius = eraserSize;
    const radiusSq = radius * radius;
    setStrokes((prev) => {
      const next = [];
      let changed = false;

      const segmentIntersectsCircle = (a, b) => {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        if (dx === 0 && dy === 0) {
          const ax = a.x - point.x;
          const ay = a.y - point.y;
          return ax * ax + ay * ay <= radiusSq;
        }
        const t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy);
        const clamped = Math.max(0, Math.min(1, t));
        const cx = a.x + clamped * dx;
        const cy = a.y + clamped * dy;
        const cxDx = cx - point.x;
        const cyDy = cy - point.y;
        return cxDx * cxDx + cyDy * cyDy <= radiusSq;
      };

      prev.forEach((stroke) => {
        let segment = [];
        let lastKept = null;
        stroke.points.forEach((pt) => {
          const dx = pt.x - point.x;
          const dy = pt.y - point.y;
          const inside = dx * dx + dy * dy <= radiusSq;
          if (inside) {
            changed = true;
            if (segment.length >= 2) {
              next.push({ points: segment, color: stroke.color });
            }
            segment = [];
            lastKept = null;
            return;
          }

          if (lastKept && segmentIntersectsCircle(lastKept, pt)) {
            changed = true;
            if (segment.length >= 2) {
              next.push({ points: segment, color: stroke.color });
            }
            segment = [pt];
            lastKept = pt;
            return;
          }

          segment.push(pt);
          lastKept = pt;
        });
        if (segment.length >= 2) {
          next.push({ points: segment, color: stroke.color });
        }
      });

      if (changed) {
        setHasPending(next.length > 0);
      }
      return next;
    });
  };

  const sendAnnotation = async () => {
    if (!frameId) return alert("No frame id yet. Start streaming first.");
    if (!strokes.length) {
      return alert("Draw at least one stroke.");
    }
    setIsSending(true);
    const annotationPayloads = strokes.map((stroke) => ({
      geometry_type: "polyline",
      points: stroke.points.map((p) => ({ x: p.x, y: p.y })),
      metadata: { color: stroke.color || color },
    }));
    const payload = {
      frame_id: frameId,
      annotations: annotationPayloads,
    };
    try {
      const res = await fetch(`${API_BASE}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      await res.json();
      setStrokes([]);
      setHasPending(false);
      drawingRef.current = false;
      currentStrokeRef.current = [];
      currentStrokeColorRef.current = color;
      alert("Annotation sent");
    } catch (err) {
      alert(`Failed to send annotation: ${err.message}`);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>TissueTrackr React Annotator</h1>
          <p>Stream frames from the backend, view landmarks, draw and send annotations.</p>
        </div>
        <div className="status">
          <span>Stream: {status}</span>
          <span>Frame: {frameId || "-"}</span>
          <span>Landmarks: {landmarks.length}</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="block">
            <h2>Stream</h2>
            <p className="muted">Connect to the backend WebSocket to receive frames.</p>
            <div className="row">
              <button onClick={connectSockets} disabled={streaming || status === "connecting"}>
                {status === "connecting" ? "Connecting..." : streaming ? "Connected" : "Connect"}
              </button>
              <button onClick={disconnectSockets} className="ghost" disabled={!streaming}>
                Disconnect
              </button>
              <button onClick={togglePlay} disabled={!streaming || hasPending}>
                {isPlaying ? "Pause" : "Play"}
              </button>
            </div>
          </div>

          <div className="block">
            <h2>Annotation</h2>
            <label className="label">Stroke color</label>
            <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
            <label className="label">Mode</label>
            <div className="row">
              <button
                type="button"
                className={mode === "draw" ? "" : "ghost"}
                onClick={() => setMode("draw")}
              >
                Draw
              </button>
              <button
                type="button"
                className={mode === "still" ? "" : "ghost"}
                onClick={() => setMode("still")}
              >
                Still
              </button>
              <button
                type="button"
                className={mode === "erase" ? "" : "ghost"}
                onClick={() => setMode("erase")}
              >
                Erase
              </button>
              <button type="button" className="ghost" onClick={eraseAll}>
                Erase all
              </button>
            </div>
            {mode === "erase" && (
              <>
                <label className="label">Eraser size</label>
                <input
                  type="range"
                  min="6"
                  max="48"
                  value={eraserSize}
                  onChange={(e) => setEraserSize(Number(e.target.value))}
                />
              </>
            )}
            <p className="muted">Draw: drag to sketch. Still: no drawing. Erase: drag over strokes to remove them.</p>
          </div>
        </section>

        <section className="viewer">
          <div className="frameShell">
            {frameSrc ? (
              <img
                ref={frameRef}
                src={frameSrc}
                alt="Live stream"
                className="frameImage"
                onLoad={() => {
                  resizeOverlay();
                  drawOverlay();
                }}
              />
            ) : (
              <div className="framePlaceholder">Waiting for stream...</div>
            )}
            <canvas
              ref={overlayRef}
              id="overlay"
              className={`overlay ${mode}`}
              style={mode === "erase" ? { cursor: eraserCursor() } : undefined}
              onClick={handleOverlayClick}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
            ></canvas>
            <div className="badge">
              {streaming ? `Frame ${frameId || "-"} • ${landmarks.length} landmarks` : "Waiting..."}
            </div>
            {hasPending && (
              <div className="actionBar">
                <button onClick={sendAnnotation} disabled={!streaming || isSending}>
                  {isSending ? "Saving..." : "Save annotation"}
                </button>
                <button className="ghost" type="button" onClick={cancelAnnotation} disabled={isSending}>
                  Cancel
                </button>
              </div>
            )}
            {showPlaybackHint && (
              <button
                type="button"
                className="playbackHint"
                onClick={(e) => {
                  e.stopPropagation();
                  togglePlay();
                }}
              >
                {lastPlaybackAction === "play" ? "❚❚" : "▶"}
              </button>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
