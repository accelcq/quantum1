# Quantum QNN Training Module
import os
import json
import numpy as np
from datetime import datetime
from typing import Any, List, Dict
from fastapi import APIRouter, Request, HTTPException
from sklearn.metrics import mean_squared_error
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from scipy.optimize import minimize
try:
    from main import (
        api_predict_quantum_simulator,
        api_predict_quantum_machine,
        fetch_and_cache_stock_data_json,
        make_features,
        save_model,
        save_train_data,
        log_step,
    )
except ImportError:
    from app.main import (
        api_predict_quantum_simulator,
        api_predict_quantum_machine,
        fetch_and_cache_stock_data_json,
        make_features,
        save_model,
        save_train_data,
        log_step,
    )
from qiskit.quantum_info import SparsePauliOp
import traceback

router = APIRouter()

try:
    from Qsimulator import router as qsimulator_router
except ImportError:
    from app.Qsimulator import router as qsimulator_router

# Quantum QNN Training (Simulator)
def train_quantum_qnn(symbols: List[str]) -> Dict[str, str]:
    log_step("TrainQuantumQNN", f"Training quantum QNN for symbols: {symbols}")
    results = {}
    for symbol in symbols:
        try:
            df = fetch_one_week_data(symbol)
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
            from qiskit_aer import AerSimulator
            from qiskit.primitives import Estimator
            backend = AerSimulator()
            log_step("QuantumML", f"Using backend: {backend.name()}")
            estimator = Estimator(backend=backend)
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
                    log_step("QuantumML", f"Estimator input types: qc={type(qc)}, observable={type(observable)}")
                    try:
                        value = estimator.run(qc, observable).result().values[0]
                    except Exception as est_e:
                        log_step("QuantumML", f"Estimator error: {str(est_e)}\n{traceback.format_exc()}")
                        raise
                    values.append(value)
                return np.mean((np.array(values) - y) ** 2)
            theta0 = np.random.rand(num_features)
            res = minimize(objective, theta0, method='COBYLA')
            save_model(res.x, f"{symbol}_quantum_qnn_params.pkl")
            save_train_data({"x_train": x.tolist(), "y_train": y.tolist()}, f"{symbol}_qnn_train_data.json")
            log_step("TrainQuantumQNN", f"Trained and saved QNN params for {symbol}")
            results[symbol] = "trained"
        except HTTPException as e:
            log_step("TrainQuantumQNN", f"HTTPException for {symbol}: {e.detail}")
            results[symbol] = f"error: {e.detail}"
        except Exception as e:
            tb = traceback.format_exc()
            log_step("TrainQuantumQNN", f"Exception for {symbol}: {str(e)}\n{tb}")
            results[symbol] = f"error: {str(e)}"
    return results

@router.post("/train/quantum")
def api_train_quantum(_request: Request) -> dict:
    log_step("API", "/train/quantum called")
    result = train_quantum_qnn(["AAPL"])  # or pass symbols as needed
    log_step("API", "Quantum QNN training complete")
    return {"status": "success", "trained": result}
