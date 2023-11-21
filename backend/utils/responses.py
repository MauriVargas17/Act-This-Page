from pydantic import BaseModel
from enum import Enum

class StatusResponse(BaseModel):
    status: str
    model_name: str

class PredictionType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    COMPARISON = "comparison"

class GeneralPrediction(BaseModel):
    prediction_type: str

class Comparison(GeneralPrediction):
    prediction_type: PredictionType = PredictionType.COMPARISON
    similarity: float
    time_ms: float
    image_1_embedding: list[int]
    image_2_embedding: list[int]
