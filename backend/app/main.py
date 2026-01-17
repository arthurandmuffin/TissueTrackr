from fastapi import FastAPI

app = FastAPI(title="HoloRay Backend")

@app.get("/health")
def health():
    return {"ok": True}
