from pydantic import BaseModel


class Prediction(BaseModel):
    label: str
    label_ko: str | None = None
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
