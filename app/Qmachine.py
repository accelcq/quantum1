# Quantum Machine Learning API for Stock Prediction using real hardware
import os
import json
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Any, List, Dict
from sklearn.metrics import mean_squared_error
from .main import fetch_and_cache_stock_data_json, make_features, quantum_predict, save_model, save_train_data, log_step
# --- Import quantum QNN training if needed ---
from .Qtraining import train_quantum_qnn

# Read IBMQ API token from environment
IBMQ_API_TOKEN = os.getenv("IBMQ_API_TOKEN")

router = APIRouter()

def get_today_str():
    return datetime.now().strftime('%Y-%m-%d')

@router.post("/predict/quantum/machine/{backend}")
def api_predict_quantum_machine(backend: str, symbols: List[str], request: Request):
    log_step("API", f"POST /predict/quantum/machine/{backend} called for symbols: {symbols}")
    today = get_today_str()
    def event_stream():
        for symbol in symbols:
            pred_cache = f"{symbol}_quantum_{backend}_pred_{today}.json"
            if os.path.exists(pred_cache):
                log_step("API", f"Returning cached quantum hardware prediction for {symbol}")
                with open(pred_cache, "r") as f:
                    yield f"data: {json.dumps({symbol: json.load(f)})}\n\n"
                continue
            df = fetch_and_cache_stock_data_json(symbol)
            x, y, dates = make_features(df)
            x_train = x[:400]
            x_test = x[400:]
            y_train = y[:400]
            y_test = y[400:]
            # Real hardware prediction
            try:
                y_pred, vqr = quantum_predict(x_train, y_train, x_test, backend_name=backend)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    "dates": dates[400:],
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "mse": mse,
                    "model_type": f"vqr_{backend}"
                }
                with open(pred_cache, "w") as f:
                    json.dump(result, f)
                save_model(vqr, f"{symbol}_quantum_{backend}_{today}.pkl")
                save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_qnn_train_data_{backend}_{today}.json")
                log_step("API", f"Quantum hardware prediction complete and cached for {symbol}")
                yield f"data: {json.dumps({symbol: result})}\n\n"
            except Exception as e:
                log_step("API", f"Quantum hardware prediction error for {symbol}: {str(e)}")
                yield f"data: {json.dumps({symbol: {'error': str(e)}})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
