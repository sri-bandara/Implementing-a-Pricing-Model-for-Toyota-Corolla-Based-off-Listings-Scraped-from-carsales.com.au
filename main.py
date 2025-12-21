import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------
# Create FastAPI app
# -----------------------
app = FastAPI(title="Corolla Price API")

# -----------------------
# Load model once
# -----------------------
model = joblib.load("price_model.joblib")
range_info = joblib.load("price_range.joblib")

LOW_Q = range_info["low_q"]
HIGH_Q = range_info["high_q"]

FEATURE_COLS = [
    "badge", "spec", "body_style",
    "kms", "seller_type", "state", "year"
]

# -----------------------
# Define expected input
# -----------------------
class CarInput(BaseModel):
    year: int
    badge: str
    spec: str
    body_style: str
    kms: float
    seller_type: str
    state: str
    listed_price: float | None = None  


# -----------------------
# Health check endpoint
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(data: CarInput):

    X = pd.DataFrame([{
        "badge": data.badge,
        "spec": data.spec,
        "body_style": data.body_style,
        "kms": data.kms,
        "seller_type": data.seller_type,
        "state": data.state,
        "year": data.year
    }], columns=FEATURE_COLS)

    pred = float(model.predict(X)[0])

    lo = round(pred + LOW_Q, -2)
    hi = round(pred + HIGH_Q, -2)
    if lo > hi:
        lo, hi = hi, lo

    response = {
        "estimated_range_low": lo,
        "estimated_range_high": hi
    }

    if data.listed_price is not None:
        p = round(data.listed_price, -2)
        if p < lo:
            response["valuation"] = "Below market price"
        elif p > hi:
            response["valuation"] = "Above market price"
        else:
            response["valuation"] = "Around market price"

    return response