from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from predict import SwimTimeModel

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://swim.phqsh.me"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Hyperparams(BaseModel):
    delta: float
    champ_p: int
    dual_p: int

class Params(BaseModel):
    times: list[float]
    target: float | None
    params: Hyperparams

@app.post("/predict")
async def predict_times(params: Params):
    model = SwimTimeModel(params.times, params.target, params.params)

    return model.prediction_model()
