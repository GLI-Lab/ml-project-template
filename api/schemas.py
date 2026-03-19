from pydantic import BaseModel


class Prediction(BaseModel):
    label: str
    label_ko: str | None = None
    confidence: float


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
