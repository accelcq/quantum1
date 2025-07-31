from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi import FastAPI, Request, Depends, HTTPException, status, Response, Body # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import logging
from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv  # type: ignore
import os, requests, json, pickle, time

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")
import numpy as np # type: ignore
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
from qiskit.circuit.library import PauliFeatureMap # type: ignore
from qiskit_machine_learning.optimizers import ADAM # type: ignore
from qiskit_aer import Aer  # type: ignore
from qiskit.circuit import Parameter # type: ignore
from qiskit_machine_learning.algorithms import VQR # type: ignore
from qiskit.circuit import ParameterVector  # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import sys
try:
    from config import load_api_keys
except ImportError:
    from app.config import load_api_keys
from qiskit.quantum_info import SparsePauliOp
from app.Qsimulator import SymbolsRequest

# Qiskit 1.0.0 compatible imports (install with pip if missing):
# pip install "qiskit==1.0.0" "qiskit-aer==0.13.3" "qiskit-ibm-runtime==0.22.0" "qiskit-machine-learning==0.7.1"

# qiskit==1.0.0
# qiskit-aer==0.13.3
# qiskit-ibm-runtime==0.22.0 #0.23.0
# qiskit-machine-learning==0.7.1 #0.8.2

# Set up logging to both file and console
log_dir = os.path.join(os.getcwd(), "app", "logs")
os.makedirs(log_dir, exist_ok=True)  # Ensures the directory exists
log_filename = os.path.join(log_dir, "execution_log.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Remove definitions of log_step, get_today_str, fetch_and_cache_stock_data_json, make_features, save_model, save_train_data
# Import them from shared.py instead
from app.shared import log_step, get_today_str, fetch_and_cache_stock_data_json, fetch_stock_data, make_features, save_model, save_train_data, api_predict_quantum_simulator, api_predict_quantum_machine, quantum_machine_predict_dict
import os
PREDICTIONS_DIR = os.path.join(os.getcwd(), "data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Load API keys from environment variables or GitHub secrets
FMP_API_KEY, IBM_CLOUD_API_KEY, IBMQ_API_TOKEN = load_api_keys()

# FastAPI app to expose logs via Swagger UI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development
        "http://localhost:3000",  # React dev server
        "http://3dc94c6e-us-south.lb.appdomain.cloud",  # Frontend deployment
        "https://3dc94c6e-us-south.lb.appdomain.cloud",  # Frontend deployment HTTPS
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# JWT Config, Generate a secure key for production use "python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY = "jS0Oq4ejgkDHTS4LVL5qM1uXDBIelvrbAuUrPsB-ZPw"  # Replace with a secure key in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy user store for demo
fake_users_db: dict[str, dict[str, Any]] = {
    "accelcq": {
        "username": "accelcq",
        "full_name": "AccelCQ User",
        "hashed_password": pwd_context.hash("password123"),
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user

class LogEntry(BaseModel):
    timestamp: str
    message: str

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict-stock-simulator")
def predict_stock(request: Request, body: dict = Body({"symbols": ["AAPL"]})):
    # Accepts: { "symbols": "AAPL,GOOGL" } or { "symbols": ["AAPL", "GOOGL"] }
    symbols = body.get("symbols", ["AAPL"])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbols:
        symbols = ["AAPL"]
    results = {}
    for symbol in symbols:
        log_step("User", f"Prediction requested for symbol: {symbol}")
        log_step("Quantum Circuit", f"Creating quantum circuit for {symbol} in qasm simulator")
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        backend = Aer.get_backend('aer_simulator')
        job = backend.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts(qc)
        prediction = "up" if counts.get('1', 0) > counts.get('0', 0) else "down"
        log_step("Prediction", f"Predicted movement for {symbol}: {prediction}")
        results[symbol] = {"prediction": prediction, "counts": counts}
    return results

@app.get("/logs", response_model=List[LogEntry])
def get_logs(current_user: dict[str, Any] = Depends(get_current_user)) -> List[LogEntry]:
    # Access current_user to avoid unused variable warning
    log_step("User", f"Prediction requested by user: {current_user.get('username', 'unknown')}")
    entries: List[LogEntry] = []
    with open(log_filename, "r") as f:
        for line in f:
            # Split only on the first space to separate timestamp and message
            ts_msg = line.strip().split(" ", 1)
            if len(ts_msg) == 2:
                ts, msg = ts_msg
                entries.append(LogEntry(timestamp=ts, message=msg)) # type: ignore

    log_step("Logs", f"Retrieved {len(entries)} log entries")
    # If no entries found, add a log entry indicating no logs
    now = datetime.now(timezone.utc)
    if not entries:
        entries.append(LogEntry(timestamp=now.isoformat(), message="No logs available."))
    else:
        # Add a log entry for the retrieval action
        entries.append(LogEntry(timestamp=now.isoformat(), message="Logs retrieved successfully."))
    # Log the action of returning log entries
    log_step("Logs", "Returning log entries")
    return entries

@app.get("/")
def root():
    return {"status": "Quantum1 Backend API is up"}
@app.get("/health")
def health_check():
    return {"status": "Quantum1 Backend API is healthy"}
@app.get("/version")
def version():
    return {"version": "1.0.0", "description": "Quantum1 Backend API for stock prediction using Qiskit"}
@app.get("/docs", include_in_schema=False)
def custom_docs():
    return {"message": "Custom Swagger UI is not available in this version. Use /docs for the default Swagger UI."}

@app.get("/test-connectivity")
async def test_connectivity():
    """Test connectivity to external APIs"""
    import requests
    test_results = {}
    
    # Test basic internet connectivity
    try:
        response = requests.get("https://httpbin.org/get", timeout=5)
        test_results["httpbin"] = {
            "status": response.status_code,
            "success": response.status_code == 200,
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        test_results["httpbin"] = {
            "status": "error",
            "success": False,
            "error": str(e)
        }
    
    # Test FMP API connectivity
    try:
        FMP_API_KEY = os.getenv("FMP_API_KEY")
        response = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={FMP_API_KEY}",
            timeout=5
        )
        test_results["fmp_api"] = {
            "status": response.status_code,
            "success": response.status_code == 200,
            "response_time": response.elapsed.total_seconds(),
            "data_length": len(response.text)
        }
    except Exception as e:
        test_results["fmp_api"] = {
            "status": "error",
            "success": False,
            "error": str(e)
        }
    
    # Test DNS resolution
    try:
        import socket
        socket.gethostbyname("financialmodelingprep.com")
        test_results["dns"] = {
            "success": True,
            "message": "DNS resolution successful"
        }
    except Exception as e:
        test_results["dns"] = {
            "success": False,
            "error": str(e)
        }
    
    return test_results

# --- new code added for stock prediction ---

# --- Data Fetching ---
# --- Authentication Dependency ---
def check_ibm_keys():
    if not IBM_CLOUD_API_KEY:
        log_step("APIKeyCheck", "IBM_CLOUD_API_KEY is empty or not set.")
    if not IBMQ_API_TOKEN:
        log_step("APIKeyCheck", "IBM_QUANTUM_API_TOKEN is empty or not set.")
    if not IBM_CLOUD_API_KEY or not IBMQ_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid IBM Cloud or Quantum API Key")
    
    if not FMP_API_KEY or FMP_API_KEY == "FMP_API_KEY":
        log_step("APIKeyCheck", "FMP_API_KEY is empty, invalid, or not set.")
        raise HTTPException(status_code=401, detail="Invalid Financial Modeling Prep API Key")

# Use the robust fetch_stock_data function from shared.py instead of the local one

# --- Data Fetching with Daily JSON Cache ---
# Use the robust fetch_and_cache_stock_data_json function from shared.py
# (imported at the top of this file)

def fetch_stock_data_custom(symbol: str, start_date: str, data_period: str, data_source: str = "Yahoo Finance") -> pd.DataFrame:
    """
    Fetch stock data with custom start date and duration
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        start_date: Start date in format "YYYY-MM-DD"
        data_period: Period like "Short-term (1-7 da)", "Medium-term (8-30 da)", "Long-term (31+ da)"
        data_source: Data source ("Yahoo Finance" or "Financial Modeling Prep")
    
    Returns:
        DataFrame with stock data
    """
    log_step("DataFetch", f"Fetching custom data for {symbol} from {start_date} for {data_period}")
    
    # Validate inputs
    if data_source not in ["Financial Modeling Prep", "Yahoo Finance"]:
        raise HTTPException(status_code=400, detail=f"Unsupported data source: {data_source}")
    
    # Parse data period to get number of days
    if "1-7" in data_period:
        days = 7
    elif "8-30" in data_period:
        days = 30
    elif "31+" in data_period:
        days = 90  # Use 90 days for long-term
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported data period: {data_period}")
    
    # Validate date format and calculate end date
    try:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=days)
        end_date = end_dt.strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {start_date}. Use YYYY-MM-DD format.")
    
    if data_source == "Financial Modeling Prep":
        # Use FMP API with date range
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_API_KEY":
            log_step("DataFetch", "FMP_API_KEY is not set or is invalid.")
            raise HTTPException(status_code=500, detail="FMP_API_KEY is not set or is invalid.")
        
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                log_step("DataFetch", f"Failed to fetch data for {symbol}, status code: {r.status_code}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch data for {symbol} from FMP API. Status: {r.status_code}")
            
            data = r.json()
            if 'historical' not in data or not data['historical']:
                log_step("DataFetch", f"No historical data found for {symbol}")
                raise HTTPException(status_code=404, detail=f"No historical data found for {symbol} in the specified date range")
            
            df = pd.json_normalize(data, 'historical', ['symbol'])
            df = df.sort_values('date').reset_index(drop=True)
            log_step("DataFetch", f"Custom DataFrame created for {symbol}, shape: {df.shape}")
            return df
            
        except requests.RequestException as e:
            log_step("DataFetch", f"Network error fetching FMP data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Network error fetching data: {str(e)}")
        except Exception as e:
            log_step("DataFetch", f"Error fetching FMP data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    else:
        # For Yahoo Finance, warn about limited functionality
        log_step("DataFetch", f"Using Yahoo Finance fallback for {symbol} - custom date range may not be applied")
        try:
            return fetch_and_cache_stock_data_json(symbol)
        except Exception as e:
            log_step("DataFetch", f"Error with Yahoo Finance fallback: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching data from Yahoo Finance fallback: {str(e)}")

# --- Model Save/Load ---
MODEL_PATH: str = "quantum1_model.pkl"
DATA_PATH: str = "quantum1_train_data.json"

def load_model(path: str = MODEL_PATH) -> Any:
    log_step("ModelLoad", f"Loading model from {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    log_step("ModelLoad", f"Model loaded from {path}")
    return model

def save_trained_model(model: Any, symbol: str, model_type: str):
    today = get_today_str()
    model_file = f"{symbol}_{model_type}_{today}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    log_step("ModelSave", f"Saved {model_type} model for {symbol} to {model_file}")

def load_trained_model(symbol: str, model_type: str):
    today = get_today_str()
    model_file = f"{symbol}_{model_type}_{today}.pkl"
    with open(model_file, "rb") as f:
        return pickle.load(f)

def load_train_data(path: str = DATA_PATH) -> dict[str, Any]:
    log_step("DataLoad", f"Loading training data from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    log_step("DataLoad", f"Training data loaded from {path}")
    return data

# --- Feature Engineering ---
from typing import Tuple, Any
from numpy.typing import NDArray

# --- Classical ML ---
# from numpy.typing import NDArray

def classical_predict(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    x_test: np.ndarray[Any, Any]
) -> Tuple[np.ndarray[Any, Any], LinearRegression]:
    log_step("ClassicalML", "Training LinearRegression model")
    model = LinearRegression()
    model.fit(x_train, y_train)
    log_step("ClassicalML", "Model trained, predicting")
    y_pred = model.predict(x_test)
    log_step("ClassicalML", "Prediction complete")
    return y_pred, model

# --- API Endpoints ---

from typing import List, Dict, Any
from fastapi import Body

@app.get("/historical-data/{symbol}")
def api_historical_data(symbol: str, request: Request) -> List[Dict[str, Any]]:
    log_step("API", f"GET /historical-data/{symbol} called")
    check_ibm_keys()
    try:
        df = fetch_and_cache_stock_data_json(symbol)
    except Exception as e:
        log_step("API", f"Historical data not found for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
    log_step("API", f"Returning historical data for {symbol}")
    records = df.to_dict(orient="records")
    return [{str(k): v for k, v in record.items()} for record in records]

class SymbolsRequest(BaseModel):
    symbols: list[str] = ["AAPL"]

@app.post("/predict/classicalML")
def api_predict_classicalML(req: SymbolsRequest, request: Request = None) -> Dict[str, Dict[str, float | list | str]]:
    symbols = req.symbols
    log_step("API", f"POST /predict/classicalML called for symbols: {symbols}")
    check_ibm_keys()
    results: dict[str, dict[str, float | list | str]] = {}
    today = get_today_str()
    for symbol in symbols:
        pred_cache = os.path.join(PREDICTIONS_DIR, f"{symbol}_classical_pred_{today}.json")
        if os.path.exists(pred_cache):
            log_step("API", f"Returning cached classical prediction for {symbol}")
            with open(pred_cache, "r") as f:
                results[symbol] = json.load(f)
            continue
        # Use daily JSON data for training
        df = fetch_and_cache_stock_data_json(symbol)
        x, y, dates = make_features(df)
        x_train = x[:400]
        x_test = x[400:]
        y_train = y[:400]
        y_test = y[400:]
        y_pred, model = classical_predict(x_train, y_train, x_test)
        mse = mean_squared_error(y_test, y_pred)
        result = {
            "dates": dates[400:],
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "mse": mse,
            "model_type": "classicalML"
        }
        results[symbol] = result
        with open(pred_cache, "w") as f:
            json.dump(result, f)
        save_model(model, f"{symbol}_classical_{today}.pkl")
        save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_classical_train_data_{today}.json")
        log_step("API", f"Classical prediction complete and cached for {symbol}")
    log_step("API", "Returning classical prediction results")
    return results

# Deprecated endpoints (for clarity)
def api_predict_classical(symbols: List[str], request: Request) -> Dict[str, Dict[str, float | List[Any] | str]]:
    pass  # Deprecated, replaced by api_predict_classicalML

def api_train_classical(_request: Request) -> dict[str, object]:
    pass  # Deprecated, replaced by api_train_classicalML

# --- Top 10 stock symbols (example, can be customized) ---
TOP_10_SYMBOLS = [
    "AAPL" # Apple Inc.
]
#, "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH"
# --- Fetch 1 week of historical data for a symbol ---
def fetch_one_week_data(symbol: str) -> pd.DataFrame:
    log_step("DataFetch", f"Fetching 1 week data for {symbol}")
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_API_KEY":
        log_step("DataFetch", "FMP_API_KEY is not set or is invalid.")
        raise HTTPException(status_code=500, detail="FMP_API_KEY is not set or is invalid.")
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=7&apikey={FMP_API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        log_step("DataFetch", f"Failed to fetch 1 week data for {symbol}, status code: {r.status_code}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch 1 week data for {symbol}")
    data = r.json()
    if 'historical' not in data or not data['historical']:
        log_step("DataFetch", f"No historical data found for {symbol}")
        raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
    df: pd.DataFrame = pd.json_normalize(data, 'historical', ['symbol'])
    df = df.sort_values('date').reset_index(drop=True)
    log_step("DataFetch", f"1 week DataFrame created for {symbol}, shape: {df.shape}")
    return df

# --- Classical ANN Training ---
from sklearn.neural_network import MLPRegressor # type: ignore

def train_classical_ann(symbols: list[str] = TOP_10_SYMBOLS) -> dict[str, str]:
    log_step("TrainClassicalANN", f"Training classical ANN for symbols: {symbols}")
    results = {}
    for symbol in symbols:
        try:
            df = fetch_one_week_data(symbol)
            if len(df) < 3:
                log_step("TrainClassicalANN", f"Not enough data for {symbol}, skipping.")
                results[symbol] = "not enough data"
                continue
            x, y, _ = make_features(df, window=2, n_points=7)
            if len(x) == 0:
                log_step("TrainClassicalANN", f"No features for {symbol}, skipping.")
                results[symbol] = "no features"
                continue
            model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
            model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
            model.fit(x, y)
            save_model(model, f"{symbol}_classical_ann.pkl")
            save_train_data({"x_train": x.tolist(), "y_train": y.tolist()}, f"{symbol}_ann_train_data.json")
            log_step("TrainClassicalANN", f"Trained and saved ANN model for {symbol}")
            results[symbol] = "trained"
        except HTTPException as e:
            log_step("TrainClassicalANN", f"HTTPException for {symbol}: {e.detail}")
            results[symbol] = f"error: {e.detail}"
        except Exception as e:
            log_step("TrainClassicalANN", f"Exception for {symbol}: {str(e)}")
            results[symbol] = f"error: {str(e)}"
    return results

# --- FastAPI Endpoints for Training ---
@app.post("/train/classicalML")
def api_train_classicalML(req: SymbolsRequest = Body({"symbols": ["AAPL"]}), request: Request = None) -> dict[str, object]:
    symbols = req.symbols
    check_ibm_keys()
    log_step("API", "/train/classicalML called")
    result = train_classical_ann(symbols)
    log_step("API", "Classical ANN training complete")
    return {"status": "success", "trained": result}

# --- Helper: Check and auto-train if model missing in prediction endpoints ---
def ensure_classical_ann(symbol: str):
    model_path = f"{symbol}_classical_ann.pkl"
    if not os.path.exists(model_path):
        log_step("AutoTrain", f"Classical ANN model missing for {symbol}, training now.")
        train_classical_ann([symbol])

# --- Import and include router for quantum training ---
try:
    from Qtraining import router as qtraining_router
except ImportError:
    from app.Qtraining import router as qtraining_router

app.include_router(qtraining_router)
# --- Import and include router for quantum simulator prediction ---
try:
    from Qsimulator import router as qsimulator_router
except ImportError:
    from app.Qsimulator import router as qsimulator_router

app.include_router(qsimulator_router)
# --- Import and include router for quantum machine prediction ---
try:
    from Qmachine import router as qmachine_router
except ImportError:
    from app.Qmachine import router as qmachine_router

app.include_router(qmachine_router)

@app.post("/predict/compare")
async def api_predict_compare(request: Request, body: dict = Body(..., example={"symbols": ["AAPL"], "backend": "ibm_brisbane"})):
    try:
        data = body if body else await request.json()
        symbols = data.get("symbols", [])
        backend = data.get("backend") or "ibm_brisbane"
        if not symbols or not isinstance(symbols, list):
            raise HTTPException(status_code=400, detail="'symbols' must be a non-empty list. Example: { 'symbols': ['AAPL'], 'backend': 'ibm_brisbane' }")
        results = {}
        for symbol in symbols:
            try:
                classical = api_predict_classical([symbol], request)
                if classical is None:
                    log_step("ERROR", f"api_predict_classical returned None for {symbol}")
                    classical = {}
                quantum_sim = api_predict_quantum_simulator(SymbolsRequest(symbols=[symbol]), request)
                if quantum_sim is None:
                    log_step("ERROR", f"api_predict_quantum_simulator returned None for {symbol}")
                    quantum_sim = {}
                quantum_real = quantum_machine_predict_dict(backend, [symbol])
                if quantum_real is None:
                    log_step("ERROR", f"quantum_machine_predict_dict returned None for {symbol}")
                    quantum_real = {}
                results[symbol] = {
                    "classical": classical.get(symbol, {}),
                    "quantum_simulator": quantum_sim.get(symbol, {}),
                    "quantum_real": quantum_real.get(symbol, {})
                }
            except Exception as e:
                log_step("ERROR", f"Exception for symbol {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}
        return results
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_step("ERROR", f"Exception in /predict/compare: {str(e)}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/quantum/machine/{backend}")
async def predict_quantum_machine(
    backend: str = "ibm_brisbane",
    payload: dict = Body(...)
):
    symbols = payload.get("symbols", [])
    days = payload.get("days", 5)
    if not symbols or not isinstance(symbols, list):
        return {"error": "Please provide a list of stock symbols as 'symbols' in the request body."}
    from app.shared import predict_quantum_machine_multi
    results = predict_quantum_machine_multi(symbols, days, backend)
    return {"status": "success", "predictions": results}

from fastapi import Body

class AdvancedPredictRequest(BaseModel):
    symbols: list[str]
    data_source: str
    start_date: str
    market: str
    data_period: str
    model_type: str

@app.post("/predict/advanced")
def predict_advanced(request: AdvancedPredictRequest, user: dict = Depends(get_current_user)):
    log_step("API", f"POST /predict/advanced called for symbols: {request.symbols}, data_source: {request.data_source}, start_date: {request.start_date}, market: {request.market}, data_period: {request.data_period}, model_type: {request.model_type}")
    check_ibm_keys()
    
    # Validate all parameters before processing
    validate_advanced_prediction_params(
        data_source=request.data_source,
        market=request.market,
        data_period=request.data_period,
        model_type=request.model_type,
        start_date=request.start_date
    )
    
    results = {}
    
    for symbol in request.symbols:
        start_time = time.time()
        try:
            log_step("API", f"Processing symbol: {symbol}")
            
            # Step 1: Collect historical data
            data_fetch_start = time.time()
            df = fetch_stock_data_custom(
                symbol=symbol,
                start_date=request.start_date,
                data_period=request.data_period,
                data_source=request.data_source
            )
            data_fetch_time = time.time() - data_fetch_start
            log_step("API", f"Fetched {len(df)} rows of data for {symbol} in {data_fetch_time:.2f}s")
            
            # Step 2: Create features from the historical data
            feature_start = time.time()
            x, y, dates = make_features(df)
            feature_time = time.time() - feature_start
            
            if len(x) < 10 or len(y) < 10:
                results[symbol] = {
                    "error": f"Not enough data for prediction. Got {len(x)} samples, need at least 10.",
                    "data_rows": len(df),
                    "features_created": len(x),
                    "data_fetch_time": data_fetch_time,
                    "feature_creation_time": feature_time
                }
                continue
            
            # Step 3: Split data for training and testing
            split_point = int(len(x) * 0.8)
            x_train = x[:split_point] if split_point > 0 else x[:-1]
            x_test = x[split_point:] if split_point < len(x) else x[-1:]
            y_train = y[:split_point] if split_point > 0 else y[:-1]
            y_test = y[split_point:] if split_point < len(y) else y[-1:]
            
            if len(x_test) == 0:
                x_test = x[-1:] 
                y_test = y[-1:]
            
            # Step 4: Train and predict based on model type
            if request.model_type == "Classical ML Mode":
                log_step("API", f"Training and predicting with Classical ML for {symbol}")
                
                # Training phase
                train_start = time.time()
                y_pred, model = classical_predict(x_train, y_train, x_test)
                train_time = time.time() - train_start
                
                # Prediction accuracy metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(mse)
                
                # Calculate R² score
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                
                # Calculate directional accuracy (up/down prediction)
                y_test_direction = np.diff(y_test) > 0
                y_pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(y_test_direction == y_pred_direction) if len(y_test_direction) > 0 else 0
                
                results[symbol] = {
                    "dates": dates[split_point:] if len(dates) > split_point else dates[-len(y_pred):],
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "model_type": "classicalML",
                    "data_period": request.data_period,
                    "data_source": request.data_source,
                    "train_samples": len(x_train),
                    "test_samples": len(x_test),
                    # Performance metrics
                    "accuracy_metrics": {
                        "mse": float(mse),
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "r2_score": float(r2),
                        "directional_accuracy": float(directional_accuracy)
                    },
                    # Timing metrics
                    "timing_metrics": {
                        "data_fetch_time": data_fetch_time,
                        "feature_creation_time": feature_time,
                        "training_time": train_time,
                        "total_time": time.time() - start_time
                    }
                }
                
            elif request.model_type == "Quantum ML Simulator":
                log_step("API", f"Training and predicting with Quantum ML Simulator for {symbol}")
                try:
                    # Training and prediction phase
                    train_start = time.time()
                    
                    # Use existing quantum simulator prediction
                    from app.shared import api_predict_quantum_simulator
                    from app.Qsimulator import SymbolsRequest
                    quantum_result = api_predict_quantum_simulator(SymbolsRequest(symbols=[symbol]), None)
                    
                    train_time = time.time() - train_start
                    
                    if quantum_result and symbol in quantum_result:
                        qresult = quantum_result[symbol]
                        y_pred = np.array(qresult.get("y_pred", []))
                        y_test_quantum = np.array(qresult.get("y_test", []))
                        
                        # Align prediction and test data lengths
                        min_len = min(len(y_pred), len(y_test_quantum))
                        if min_len > 0:
                            y_pred = y_pred[:min_len]
                            y_test_quantum = y_test_quantum[:min_len]
                            
                            # Calculate accuracy metrics
                            mse = mean_squared_error(y_test_quantum, y_pred)
                            mae = np.mean(np.abs(y_test_quantum - y_pred))
                            rmse = np.sqrt(mse)
                            
                            from sklearn.metrics import r2_score
                            r2 = r2_score(y_test_quantum, y_pred)
                            
                            # Directional accuracy
                            y_test_direction = np.diff(y_test_quantum) > 0 if len(y_test_quantum) > 1 else []
                            y_pred_direction = np.diff(y_pred) > 0 if len(y_pred) > 1 else []
                            directional_accuracy = np.mean(y_test_direction == y_pred_direction) if len(y_test_direction) > 0 else 0
                            
                            results[symbol] = {
                                **qresult,
                                "model_type": "quantumML_simulator",
                                "data_period": request.data_period,
                                "data_source": request.data_source,
                                "quantum_backend": "aer_simulator",
                                "accuracy_metrics": {
                                    "mse": float(mse),
                                    "mae": float(mae),
                                    "rmse": float(rmse),
                                    "r2_score": float(r2),
                                    "directional_accuracy": float(directional_accuracy)
                                },
                                "timing_metrics": {
                                    "data_fetch_time": data_fetch_time,
                                    "feature_creation_time": feature_time,
                                    "training_time": train_time,
                                    "total_time": time.time() - start_time
                                }
                            }
                        else:
                            results[symbol] = {"error": "No valid prediction data from quantum simulator"}
                    else:
                        results[symbol] = {"error": "Quantum simulator prediction failed"}
                        
                except Exception as e:
                    log_step("ERROR", f"Quantum simulator prediction failed for {symbol}: {str(e)}")
                    results[symbol] = {"error": f"Quantum simulator prediction failed: {str(e)}"}
                    
            elif request.model_type == "Quantum ML Real Machine":
                log_step("API", f"Training and predicting with Quantum ML Real Machine for {symbol}")
                try:
                    # Training and prediction phase
                    train_start = time.time()
                    
                    # Use quantum real machine prediction
                    from app.shared import quantum_machine_predict_dict
                    quantum_result = quantum_machine_predict_dict("ibm_brisbane", [symbol])
                    
                    train_time = time.time() - train_start
                    
                    if quantum_result and symbol in quantum_result:
                        qresult = quantum_result[symbol]
                        y_pred = np.array(qresult.get("y_pred", []))
                        
                        # Use last few data points as test data for comparison
                        test_days = min(5, len(y))
                        y_test_quantum = y[-test_days:] if test_days > 0 else []
                        test_dates = dates[-test_days:] if test_days > 0 and len(dates) >= test_days else []
                        
                        if len(y_test_quantum) > 0 and len(y_pred) > 0:
                            # Align lengths
                            min_len = min(len(y_pred), len(y_test_quantum))
                            y_pred_aligned = y_pred[:min_len]
                            y_test_aligned = y_test_quantum[:min_len]
                            
                            # Calculate accuracy metrics
                            mse = mean_squared_error(y_test_aligned, y_pred_aligned)
                            mae = np.mean(np.abs(y_test_aligned - y_pred_aligned))
                            rmse = np.sqrt(mse)
                            
                            from sklearn.metrics import r2_score
                            r2 = r2_score(y_test_aligned, y_pred_aligned)
                            
                            # Directional accuracy
                            y_test_direction = np.diff(y_test_aligned) > 0 if len(y_test_aligned) > 1 else []
                            y_pred_direction = np.diff(y_pred_aligned) > 0 if len(y_pred_aligned) > 1 else []
                            directional_accuracy = np.mean(y_test_direction == y_pred_direction) if len(y_test_direction) > 0 else 0
                            
                            results[symbol] = {
                                "dates": test_dates[:min_len],
                                "y_test": y_test_aligned.tolist(),
                                "y_pred": y_pred_aligned.tolist(),
                                "model_type": "quantumML_real_machine",
                                "data_period": request.data_period,
                                "data_source": request.data_source,
                                "quantum_backend": "ibm_brisbane",
                                "prediction_days": test_days,
                                "accuracy_metrics": {
                                    "mse": float(mse),
                                    "mae": float(mae),
                                    "rmse": float(rmse),
                                    "r2_score": float(r2),
                                    "directional_accuracy": float(directional_accuracy)
                                },
                                "timing_metrics": {
                                    "data_fetch_time": data_fetch_time,
                                    "feature_creation_time": feature_time,
                                    "training_time": train_time,
                                    "total_time": time.time() - start_time
                                }
                            }
                        else:
                            results[symbol] = {"error": "No valid data for quantum real machine prediction"}
                    else:
                        results[symbol] = {"error": "Quantum real machine prediction failed"}
                        
                except Exception as e:
                    log_step("ERROR", f"Quantum real machine prediction failed for {symbol}: {str(e)}")
                    results[symbol] = {"error": f"Quantum real machine prediction failed: {str(e)}"}
                    
            else:
                results[symbol] = {"error": f"Model type '{request.model_type}' not implemented. Available: 'Classical ML Mode', 'Quantum ML Simulator', 'Quantum ML Real Machine'"}
                
        except Exception as e:
            log_step("ERROR", f"Error in advanced prediction for {symbol}: {str(e)}")
            results[symbol] = {
                "error": str(e),
                "timing_metrics": {
                    "total_time": time.time() - start_time
                }
            }
    
    return {"status": "success", "results": results}

# --- Data Validation Functions ---
def validate_advanced_prediction_params(data_source: str, market: str, data_period: str, model_type: str, start_date: str):
    """Validate parameters for advanced prediction"""
    
    # Validate data source
    supported_data_sources = ["Financial Modeling Prep", "Yahoo Finance"]
    if data_source not in supported_data_sources:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported data source '{data_source}'. Supported sources: {', '.join(supported_data_sources)}"
        )
    
    # Validate market
    supported_markets = ["US - United States", "EU - Europe", "IN - India"]
    if market not in supported_markets:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported market '{market}'. Supported markets: {', '.join(supported_markets)}"
        )
    
    # Validate data period
    supported_periods = ["Short-term (1-7 da)", "Medium-term (8-30 da)", "Long-term (31+ da)"]
    if data_period not in supported_periods:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported data period '{data_period}'. Supported periods: {', '.join(supported_periods)}"
        )
    
    # Validate model type
    supported_models = ["Classical ML Mode", "Quantum ML Simulator", "Quantum ML Real Machine"]
    if model_type not in supported_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model type '{model_type}'. Supported models: {', '.join(supported_models)}"
        )
    
    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid date format: {start_date}. Use YYYY-MM-DD format."
        )
    
    # Special warnings
    if data_source == "Yahoo Finance":
        log_step("API", f"Warning: Yahoo Finance fallback - custom date range may not be applied")
    
    if market != "US - United States":
        log_step("API", f"Warning: Market '{market}' may have limited stock symbol support")
