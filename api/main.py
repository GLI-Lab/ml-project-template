import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from api.model import model_managers
from api.schemas import HealthResponse, PredictResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_manager_map = {m.name: m for m in model_managers}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for manager in model_managers:
        logger.info("Loading model: %s", manager.name)
        manager.load()
        logger.info("Model loaded: %s", manager.name)
    yield
    for manager in model_managers:
        logger.info("Unloading model: %s", manager.name)
        manager.unload()


app = FastAPI(
    title="ML Inference API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded={m.name: m.is_loaded for m in model_managers},
    )


@app.get("/api/models")
def list_models():
    return {"models": list(_manager_map.keys())}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile, model: str = "resnet50"):
    manager = _manager_map.get(model)
    if manager is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(_manager_map.keys())}",
        )
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model}' is not loaded")

    image_bytes = await file.read()
    logger.info("Received prediction request: model=%s filename=%s size=%d",
                model, file.filename, len(image_bytes))
    try:
        predictions = manager.predict_image(image_bytes)
    except Exception as e:
            logger.exception("Prediction failed for model: %s", model)
            raise HTTPException(status_code=500, detail=f"Prediction failed for model: {model} — {e}")

    logger.info("[%s] Top prediction: %s (%.4f)", model, predictions[0]["label"], predictions[0]["confidence"])
    return PredictResponse(model=model, predictions=predictions)


@app.post("/api/reload")
def reload():
    for manager in model_managers:
        logger.info("Reloading model: %s", manager.name)
        manager.unload()
        manager.load()
        logger.info("Model reloaded: %s", manager.name)
    return {"status": "reloaded"}
