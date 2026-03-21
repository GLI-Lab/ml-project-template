import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from api.models import model_managers
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
    return {
        "models": [
            {
                "name": m.name,
                "loaded": m.is_loaded,
                "config": m.get_config(),
            }
            for m in model_managers
        ],
    }


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


_MODELS_DIR = Path(__file__).parent.parent / "models"
_WEIGHT_EXTS = {".pt", ".pth"}


@app.get("/api/browse")
def browse(path: str = ""):
    """models/ 디렉토리를 탐색하여 하위 폴더와 .pt/.pth 파일 목록을 반환합니다."""
    base = _MODELS_DIR
    target = (base / path).resolve()

    # models/ 밖으로 나가는 경로 차단
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    dirs = sorted(d.name for d in target.iterdir() if d.is_dir())
    weights = sorted(f.name for f in target.iterdir() if f.is_file() and f.suffix in _WEIGHT_EXTS)
    rel_path = str(target.relative_to(base))

    return {
        "path": rel_path,
        "dirs": dirs,
        "weights": weights,
        "num_weights": len(weights),
    }


@app.post("/api/load-weights")
def load_weights(model: str, path: str):
    """선택한 모델에 .pt/.pth 가중치 파일을 로드합니다."""
    manager = _manager_map.get(model)
    if manager is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(_manager_map.keys())}",
        )
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model}' is not loaded")

    weights_path = Path(path).resolve()
    if not str(weights_path).startswith(str(_MODELS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not weights_path.is_file():
        raise HTTPException(status_code=404, detail="Weight file not found")

    try:
        manager.load_weights(weights_path)
        logger.info("Loaded weights for model '%s' from %s", model, weights_path)
    except Exception as e:
        logger.exception("Failed to load weights for model '%s'", model)
        raise HTTPException(status_code=500, detail=f"Failed to load weights: {e}")

    return {"status": "loaded", "model": model, "path": str(weights_path)}


@app.post("/api/reload")
def reload(model: str | None = None):
    """모델을 기본 pretrained 가중치로 다시 로드합니다. model 미지정 시 전체 리로드."""
    targets = [_manager_map[model]] if model else model_managers
    if model and model not in _manager_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(_manager_map.keys())}",
        )
    for manager in targets:
        logger.info("Reloading model: %s", manager.name)
        manager.unload()
        manager.load()
        logger.info("Model reloaded: %s", manager.name)
    return {"status": "reloaded", "models": [m.name for m in targets]}
