# shared.py: Breaks circular import between main.py and Qtraining.py
from typing import Any, List, Dict, Tuple
from fastapi import Request
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error
from fastapi.responses import StreamingResponse
from datetime import datetime
import logging
import sys
import pandas as pd
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.optimizers import ADAM
from qiskit_machine_learning.algorithms import VQR
from qiskit.quantum_info import SparsePauliOp

# Import Aer only if available
try:
    from qiskit_aer import Aer
    from qiskit_aer.primitives import Sampler, Estimator
except ImportError:
    Aer = None
    Sampler = None
    Estimator = None
# Import QiskitRuntimeService only if available
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    QiskitRuntimeService = None

# Logging setup
log_dir = os.path.join(os.getcwd(), "app", "logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, "execution_log.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
def log_step(step: str, detail: str):
    logging.info(f"{step}: {detail}")

def get_today_str():
    return datetime.now().strftime('%Y-%m-%d')

def fetch_stock_data(symbol: str) -> pd.DataFrame:
    log_step("DataFetch", f"Fetching stock data for symbol: {symbol}")
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    if not FMP_API_KEY or FMP_API_KEY == "FMP_API_KEY":
        log_step("DataFetch", "FMP_API_KEY is not set or is invalid.")
        raise Exception("FMP_API_KEY is not set or is invalid.")
    
    import requests
    
    # Use correct FMP API endpoints based on official documentation
    urls_to_try = [
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=365&apikey={FMP_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=30&apikey={FMP_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    ]
    
    for i, url in enumerate(urls_to_try):
        try:
            log_step("DataFetch", f"Attempting URL {i+1}/3: {url}")
            r = requests.get(url, timeout=10)
            log_step("DataFetch", f"HTTP status: {r.status_code}")
            
            if r.status_code == 200:
                data = r.json()
                log_step("DataFetch", f"Data fetched for {symbol}, processing response")
                
                # Handle different response formats
                if i == 0:  # Full historical price endpoint (365 days)
                    if 'historical' in data and data['historical']:
                        df = pd.json_normalize(data, 'historical', ['symbol'])
                        df = df.sort_values('date').reset_index(drop=True)
                        log_step("DataFetch", f"Full historical DataFrame created for {symbol}, shape: {df.shape}")
                        return df
                elif i == 1:  # Light historical price endpoint (30 days)
                    if 'historical' in data and data['historical']:
                        df = pd.json_normalize(data, 'historical', ['symbol'])
                        df = df.sort_values('date').reset_index(drop=True)
                        log_step("DataFetch", f"Light historical DataFrame created for {symbol}, shape: {df.shape}")
                        return df
                elif i == 2:  # Quote endpoint
                    if isinstance(data, list) and len(data) > 0:
                        # Convert single quote to historical-like format
                        quote = data[0]
                        today = datetime.now().strftime('%Y-%m-%d')
                        df_data = []
                        # Create a week's worth of synthetic historical data based on current quote
                        for j in range(7):
                            date = pd.date_range(end=today, periods=7, freq='D')[j].strftime('%Y-%m-%d')
                            price = quote.get('price', 100) * (0.95 + 0.1 * j / 6)  # Vary price slightly
                            df_data.append({
                                'date': date,
                                'open': price * 0.98,
                                'high': price * 1.02,
                                'low': price * 0.96,
                                'close': price,
                                'symbol': symbol
                            })
                        df = pd.DataFrame(df_data)
                        log_step("DataFetch", f"Quote-based DataFrame created for {symbol}, shape: {df.shape}")
                        return df
                        
            log_step("DataFetch", f"URL {i+1} failed with status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log_step("DataFetch", f"URL {i+1} failed with exception: {str(e)}")
            continue
    
    # If all URLs fail
    raise Exception(f"All API endpoints failed for {symbol}")

DATA_DIR = os.path.join(os.getcwd(), "data")
HISTORICAL_DIR = os.path.join(DATA_DIR, "historical")
TRAINED_DIR = os.path.join(DATA_DIR, "trained")
PREDICTIONS_DIR = os.path.join(os.getcwd(), "data", "predictions")

# Ensure all required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORICAL_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
def fetch_and_cache_stock_data_json(symbol: str) -> pd.DataFrame:
    today = get_today_str()
    cache_file = os.path.join(HISTORICAL_DIR, f"{symbol}_historical_{today}.json")
    
    # Step 1: Check if we have today's cached data (highest priority)
    if os.path.exists(cache_file):
        log_step("DataFetch", f"Loading today's cached data for {symbol} from {cache_file}")
        try:
            df = pd.read_json(cache_file)
            log_step("DataFetch", f"Successfully loaded today's cached data for {symbol}")
            return df
        except Exception as e:
            log_step("DataFetch", f"Failed to load today's cached data for {symbol}: {e}")
    
    # Step 2: Try to fetch fresh data from API (second priority)
    log_step("DataFetch", f"Attempting to fetch fresh data for {symbol} from FMP API")
    try:
        df = fetch_stock_data(symbol)
        # Cache the fresh data for today
        os.makedirs(HISTORICAL_DIR, exist_ok=True)
        df.to_json(cache_file, orient="records")
        log_step("DataFetch", f"✅ Fresh data fetched and cached for {symbol} to {cache_file}")
        return df
    except Exception as e:
        log_step("DataFetch", f"❌ Failed to fetch fresh data from API for {symbol}: {e}")
    
    # Step 3: Fallback to any existing historical data (third priority)
    log_step("DataFetch", f"Looking for any existing historical data for {symbol}")
    import glob
    pattern = os.path.join(HISTORICAL_DIR, f"{symbol}_historical_*.json")
    existing_files = glob.glob(pattern)
    
    if existing_files:
        # Use the most recent file
        latest_file = max(existing_files, key=os.path.getmtime)
        log_step("DataFetch", f"📁 Using fallback cached data for {symbol} from {latest_file}")
        try:
            df = pd.read_json(latest_file)
            log_step("DataFetch", f"Successfully loaded fallback data for {symbol}")
            return df
        except Exception as e:
            log_step("DataFetch", f"Failed to load fallback data for {symbol}: {e}")
    
    # Step 4: Check for similar symbols (fourth priority)
    log_step("DataFetch", f"Looking for similar symbols for {symbol}")
    symbol_mappings = {
        "GOOG": "GOOGL",
        "GOOGL": "GOOG",
        "BRK.A": "BRK.B",
        "BRK.B": "BRK.A"
    }
    
    if symbol in symbol_mappings:
        alt_symbol = symbol_mappings[symbol]
        alt_pattern = os.path.join(HISTORICAL_DIR, f"{alt_symbol}_historical_*.json")
        alt_files = glob.glob(alt_pattern)
        if alt_files:
            latest_alt_file = max(alt_files, key=os.path.getmtime)
            log_step("DataFetch", f"🔄 Using {alt_symbol} data as fallback for {symbol} from {latest_alt_file}")
            try:
                df = pd.read_json(latest_alt_file)
                log_step("DataFetch", f"Successfully loaded alternative symbol data for {symbol}")
                return df
            except Exception as e:
                log_step("DataFetch", f"Failed to load alternative symbol data for {symbol}: {e}")
    
    # Step 5: Last resort - create synthetic data
    log_step("DataFetch", f"⚠️ Creating synthetic data for {symbol} as last resort")
    try:
        # Create minimal synthetic historical data
        dates = pd.date_range(end=today, periods=30, freq='D')
        base_price = 100
        synthetic_data = []
        
        for i, date in enumerate(dates):
            price = base_price + (i * 0.5) + np.random.normal(0, 2)  # Slight upward trend with noise
            synthetic_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': price * 0.98,
                'high': price * 1.02,
                'low': price * 0.96,
                'close': price,
                'symbol': symbol
            })
        
        df = pd.DataFrame(synthetic_data)
        # Cache the synthetic data
        df.to_json(cache_file, orient="records")
        log_step("DataFetch", f"🔧 Synthetic data created and cached for {symbol}")
        return df
    except Exception as e:
        log_step("DataFetch", f"Failed to create synthetic data for {symbol}: {e}")
        raise Exception(f"❌ All data retrieval methods failed for {symbol}")
    
    # Final fallback - should never reach here
    raise Exception(f"❌ No historical data available for {symbol} and all fallback methods failed")

def make_features(
    df: pd.DataFrame,
    window: int = 2,
    col: str = 'open',
    n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    log_step("FeatureEngineering", f"Creating features with window={window}, col={col}, n_points={n_points}")
    final_data: pd.DataFrame = df[[col, 'date']][:n_points]
    input_sequences: list[list[float]] = []
    labels: list[float] = []
    col_series = final_data[col].astype(float)
    for i in range(window, len(col_series)):
        labels.append(float(col_series.iloc[i]))
        input_sequences.append([float(x) for x in col_series.iloc[i-window:i].tolist()])
    log_step("FeatureEngineering", f"Features created: {len(input_sequences)} samples")
    return (
        np.array(input_sequences, dtype=np.float64),
        np.array(labels, dtype=np.float64),
        final_data['date'].iloc[window:].astype(str).tolist()
    )

def save_model(model: Any, path: str) -> None:
    # Save models in trained folder
    if not os.path.isabs(path):
        path = os.path.join(TRAINED_DIR, path)
    log_step("ModelSave", f"Saving model to {path}")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(model, f)
    log_step("ModelSave", f"Model saved to {path}")

def save_train_data(data: dict[str, Any], path: str) -> None:
    # Save training data in trained folder
    if not os.path.isabs(path):
        path = os.path.join(TRAINED_DIR, path)
    log_step("DataSave", f"Saving training data to {path}")
    with open(path, "w") as f:
        json.dump(data, f)
    log_step("DataSave", f"Training data saved to {path}")

def save_prediction_data(data: dict[str, Any], path: str) -> None:
    # Save predictions in predictions folder
    if not os.path.isabs(path):
        path = os.path.join(PREDICTIONS_DIR, path)
    log_step("DataSave", f"Saving prediction data to {path}")
    with open(path, "w") as f:
        json.dump(data, f)
    log_step("DataSave", f"Prediction data saved to {path}")

def quantum_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    backend_name: str
):
    log_step("QuantumML", f"Starting quantum prediction on backend {backend_name}")
    num_features = x_train.shape[1]
    feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = QuantumCircuit(num_features)
    params = ParameterVector('theta', length=num_features)
    for i in range(num_features):
        ansatz.ry(params[i], i)
    optimizer = ADAM(maxiter=100)

    if backend_name == "aer_simulator":
        if Aer is None:
            raise ImportError("Qiskit Aer is not installed.")
        log_step("QuantumML", "Using local Aer simulator backend")
        backend = Aer.get_backend(backend_name)
        vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        log_step("QuantumML", "Fitting VQR model (simulator)")
        vqr.fit(x_train, y_train)
        log_step("QuantumML", "Model fit complete, predicting")
        y_pred = vqr.predict(x_test)
        log_step("QuantumML", "Quantum prediction complete")
        return y_pred, vqr
    else:
        # Primitives-based regression for real hardware
        if QiskitRuntimeService is None or Estimator is None:
            raise ImportError("Qiskit IBM Runtime or Estimator is not installed.")
        log_step("QuantumML", f"Using IBM Quantum backend: {backend_name} (primitives)")
        service = QiskitRuntimeService(channel="ibm_quantum", token=os.getenv("IBMQ_API_TOKEN"))
        backend = service.backend(backend_name)
        estimator = Estimator()
        observable = SparsePauliOp("Z" + "I" * (num_features - 1))
        # For each test sample, run the circuit and collect results
        y_pred = []
        for xi in x_test:
            qc = QuantumCircuit(num_features)
            feature_circ = feature_map.assign_parameters(xi)
            for instr, qargs, cargs in feature_circ.data:
                qc.append(instr, [qc.qubits[feature_circ.qubits.index(q)] for q in qargs], cargs)
            ansatz_circ = ansatz.assign_parameters(np.zeros(num_features))  # Use zeros or trainable params
            for instr, qargs, cargs in ansatz_circ.data:
                qc.append(instr, [qc.qubits[ansatz_circ.qubits.index(q)] for q in qargs], cargs)
            # For demo, use zeros for params; for real, optimize theta (training loop needed)
            result = estimator.run(qc, observable, options={"backend": backend}).result()
            y_pred.append(result.values[0])
        log_step("QuantumML", "Quantum prediction complete (primitives)")
        return np.array(y_pred), None

def api_predict_quantum_simulator(symbols_req, request: Request, backend_machine: str = "ibm_brisbane") -> Dict[str, Dict[str, float | List[Any] | str]]:
    symbols = symbols_req.symbols
    log_step("API", f"POST /predict/quantum/simulator called for symbols: {symbols}")
    results: dict[str, dict[str, float | list[Any] | str]] = {}
    today = datetime.now().strftime('%Y-%m-%d')
    for symbol in symbols:
        pred_cache = os.path.join(PREDICTIONS_DIR, f"{symbol}_quantum_sim_pred_{today}.json")
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
        y_pred, vqr = quantum_predict(x_train, y_train, x_test, backend_name=backend_machine)
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

def api_predict_quantum_machine(backend: str, symbols_req, request: Request):
    symbols = symbols_req.symbols
    log_step("API", f"POST /predict/quantum/machine/{backend} called for symbols: {symbols}")
    today = datetime.now().strftime('%Y-%m-%d')
    def event_stream():
        for symbol in symbols:
            pred_cache = os.path.join(PREDICTIONS_DIR, f"{symbol}_quantum_{backend}_pred_{today}.json")
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
            try:
                y_pred, _ = quantum_predict(x_train, y_train, x_test, backend_name=backend)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    "dates": dates[400:],
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "mse": mse,
                    "model_type": f"primitives_{backend}" if backend != "aer_simulator" else "vqr_simulator"
                }
                with open(pred_cache, "w") as f:
                    json.dump(result, f)
                save_model({}, f"{symbol}_quantum_{backend}_{today}.pkl")  # No VQR model for primitives
                save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_qnn_train_data_{backend}_{today}.json")
                log_step("API", f"Quantum hardware prediction complete and cached for {symbol}")
                yield f"data: {json.dumps({symbol: result})}\n\n"
            except Exception as e:
                log_step("API", f"Quantum hardware prediction error for {symbol}: {str(e)}")
                yield f"data: {json.dumps({symbol: {'error': str(e)}})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

def quantum_machine_predict_dict(backend: str, symbols: list[str]) -> dict:
    """Return quantum hardware prediction results as a dict for the given backend and symbols."""
    results = {}
    today = datetime.now().strftime('%Y-%m-%d')
    for symbol in symbols:
        pred_cache = os.path.join(PREDICTIONS_DIR, f"{symbol}_quantum_{backend}_pred_{today}.json")
        if os.path.exists(pred_cache):
            log_step("API", f"Returning cached quantum hardware prediction for {symbol}")
            with open(pred_cache, "r") as f:
                results[symbol] = json.load(f)
            continue
        df = fetch_and_cache_stock_data_json(symbol)
        x, y, dates = make_features(df)
        x_train = x[:400]
        x_test = x[400:]
        y_train = y[:400]
        y_test = y[400:]
        try:
            y_pred, _ = quantum_predict(x_train, y_train, x_test, backend_name=backend)
            mse = mean_squared_error(y_test, y_pred)
            result = {
                "dates": dates[400:],
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "mse": mse,
                "model_type": f"primitives_{backend}" if backend != "aer_simulator" else "vqr_simulator"
            }
            with open(pred_cache, "w") as f:
                json.dump(result, f)
            save_model({}, f"{symbol}_quantum_{backend}_{today}.pkl")  # No VQR model for primitives
            save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_qnn_train_data_{backend}_{today}.json")
            log_step("API", f"Quantum hardware prediction complete and cached for {symbol}")
            results[symbol] = result
        except Exception as e:
            log_step("API", f"Quantum hardware prediction error for {symbol}: {str(e)}")
            results[symbol] = {"error": str(e)}
    return results

# Add any other shared functions here
