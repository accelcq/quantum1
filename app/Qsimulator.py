# Quantum Machine Learning API for Stock Prediction using QASM simulator
import os
import json
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Any, List, Dict
from sklearn.metrics import mean_squared_error
from .quantum_utils import quantum_predict
from .main import fetch_and_cache_stock_data_json, make_features, save_model, save_train_data, log_step
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pydantic import BaseModel

# --- Use IBMQ_API_TOKEN from environment ---
IBMQ_API_TOKEN = os.getenv("IBMQ_API_TOKEN")
print(f"Using IBMQ_API_TOKEN: {IBMQ_API_TOKEN is not None}")

router = APIRouter()

def get_today_str():
    return datetime.now().strftime('%Y-%m-%d')

def build_ansatz(num_qubits: int, depth: int = 1) -> QuantumCircuit:
    """
    Build a generic Ry ansatz circuit with optional depth.
    Each layer applies Ry rotations to all qubits, followed by a ring of CNOTs.
    """
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        params = [Parameter(f"theta_{d}_{i}") for i in range(num_qubits)]
        for i in range(num_qubits):
            qc.ry(params[i], i)
        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)
    return qc

class SymbolsRequest(BaseModel):
    symbols: List[str]

@router.post("/predict/quantum/simulator")
def api_predict_quantum_simulator(symbols_req: SymbolsRequest, request: Request) -> Dict[str, Dict[str, float | List[Any] | str]]:
    symbols = symbols_req.symbols
    log_step("API", f"POST /predict/quantum/simulator called for symbols: {symbols}")
    results: dict[str, dict[str, float | list[Any] | str]] = {}
    today = get_today_str()
    for symbol in symbols:
        pred_cache = f"{symbol}_quantum_sim_pred_{today}.json"
        if os.path.exists(pred_cache):
            log_step("API", f"Returning cached quantum simulator prediction for {symbol}")
            with open(pred_cache, "r") as f:
                results[symbol] = json.load(f)
            continue
        df = fetch_and_cache_stock_data_json(symbol)
        x, y, dates = make_features(df)
        x_train = x[:400]
        x_test = x[400:]
        y_train = y[:400]
        y_test = y[400:]
        # Simulator prediction (AerSimulator)
        y_pred, vqr = quantum_predict(x_train, y_train, x_test, backend_name="aer_simulator")
        mse = mean_squared_error(y_test, y_pred)
        result = {
            "dates": dates[400:],
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "mse": mse,
            "model_type": "vqr_simulator"
        }
        results[symbol] = result
        with open(pred_cache, "w") as f:
            json.dump(result, f)
        save_model(vqr, f"{symbol}_quantum_sim_{today}.pkl")
        save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_qnn_train_data_sim_{today}.json")
        log_step("API", f"Quantum simulator prediction complete and cached for {symbol}")
    log_step("API", "Returning quantum simulator prediction results")
    return results
