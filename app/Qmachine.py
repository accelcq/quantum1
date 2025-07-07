# Quantum Machine Learning API for Stock Prediction using real hardware
import os
import json
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List
from sklearn.metrics import mean_squared_error
from pydantic import BaseModel

from app.shared import fetch_and_cache_stock_data_json, make_features, save_model, save_train_data, log_step

from app.shared import api_predict_quantum_machine
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# --- Use IBMQ_API_TOKEN from environment ---
IBMQ_API_TOKEN = os.getenv("IBMQ_API_TOKEN")
print(f"Using IBMQ_API_TOKEN: {IBMQ_API_TOKEN is not None}")

router = APIRouter()

def get_today_str():
    return datetime.now().strftime('%Y-%m-%d')

class SymbolsRequest(BaseModel):
    symbols: List[str]

@router.post("/predict/quantum/machine/{backend}")
def predict_quantum_machine(backend: str, symbols_req: SymbolsRequest, request: Request):
    # Return JSON dict for dashboard/frontend compatibility
    from app.shared import quantum_machine_predict_dict
    return quantum_machine_predict_dict(backend, symbols_req.symbols)

@router.post("/train/quantum/machine/{backend}")
def train_quantum_machine_backend(backend: str = "ibm_brisbane", symbols_req: SymbolsRequest = None, request: Request = None):
    """
    Train a quantum QNN on a real IBM backend (default: ibm_brisbane).
    Accepts POST body: {"symbols": ["AAPL", ...]}
    """
    log_step("API", f"/train/quantum/machine/{backend} called")
    results = {}
    symbols = symbols_req.symbols if symbols_req else ["AAPL"]
    for symbol in symbols:
        try:
            df = fetch_and_cache_stock_data_json(symbol)
            if len(df) < 3:
                log_step("TrainQuantumQNN", f"Not enough data for {symbol}, skipping.")
                results[symbol] = "not enough data"
                continue
            x, y, dates = make_features(df, window=2, n_points=7)
            if len(x) == 0:
                log_step("TrainQuantumQNN", f"No features for {symbol}, skipping.")
                results[symbol] = "no features"
                continue
            num_features = x.shape[1]
            feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
            ansatz = QuantumCircuit(num_features)
            params = ParameterVector('theta', length=num_features)
            for i in range(num_features):
                ansatz.ry(params[i], i)
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit.primitives import Estimator
            # Use the new IBM Quantum Platform authentication
            service = QiskitRuntimeService(channel="ibm_quantum", token=IBMQ_API_TOKEN)
            # Get the backend through the service
            backend_instance = service.backend(backend)
            estimator = Estimator(options={"backend": backend_instance})
            observable = SparsePauliOp("Z" + "I" * (num_features - 1))
            def objective(theta):
                values = []
                for xi in x:
                    qc = QuantumCircuit(num_features)
                    feature_map_local = PauliFeatureMap(feature_dimension=num_features, reps=1)
                    feature_circ = feature_map_local.assign_parameters(xi)
                    for instr, qargs, cargs in feature_circ.data:
                        qc.append(instr, [qc.qubits[feature_circ.qubits.index(q)] for q in qargs], cargs)
                    ansatz_circ = ansatz.assign_parameters(theta)
                    for instr, qargs, cargs in ansatz_circ.data:
                        qc.append(instr, [qc.qubits[ansatz_circ.qubits.index(q)] for q in qargs], cargs)
                    try:
                        result = estimator.run(qc, observable).result()
                        value = result.values[0]
                    except Exception as est_e:
                        log_step("QuantumML", f"Estimator error: {str(est_e)}\n{traceback.format_exc()}")
                        raise
                    values.append(value)
                return np.mean((np.array(values) - y) ** 2)
            theta0 = np.random.rand(num_features)
            from scipy.optimize import minimize
            res = minimize(objective, theta0, method='COBYLA')
            save_model(res.x, f"{symbol}_quantum_qnn_params_{backend}.pkl")
            save_train_data({"x_train": x.tolist(), "y_train": y.tolist()}, f"{symbol}_qnn_train_data_{backend}.json")
            log_step("TrainQuantumQNN", f"Trained and saved QNN params for {symbol} on backend {backend}")
            results[symbol] = "trained"
        except HTTPException as e:
            log_step("TrainQuantumQNN", f"HTTPException for {symbol}: {e.detail}")
            results[symbol] = f"error: {e.detail}"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_step("TrainQuantumQNN", f"Exception for {symbol}: {str(e)}\n{tb}")
            results[symbol] = f"error: {str(e)}"
    return {"status": "success", "trained": results}
