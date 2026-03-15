import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from api.model import model_manager
from api.schemas import HealthResponse, PredictResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model_manager.load()
    logger.info("Model loaded.")
    yield
    logger.info("Unloading model...")
    model_manager.unload()


app = FastAPI(
    title="ML Inference API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model_manager.is_loaded)


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    logger.info("Received prediction request: filename=%s size=%d", file.filename, len(image_bytes))
    try:
        predictions = model_manager.predict_image(image_bytes)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Top prediction: %s (%.4f)", predictions[0]["label"], predictions[0]["confidence"])
    return PredictResponse(predictions=predictions)


@app.post("/api/reload")
def reload():
    logger.info("Reloading model...")
    model_manager.unload()
    model_manager.load()
    logger.info("Model reloaded.")
    return {"status": "reloaded"}
