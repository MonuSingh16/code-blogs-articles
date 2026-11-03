from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)

class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class BatchRequest(BaseModel):
    items: list[IrisFeatures] = Field(min_length=1, max_length=1000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
