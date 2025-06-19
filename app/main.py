from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi import FastAPI, Request, Depends, HTTPException, status, Response # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi import FastAPI, Request, Depends, HTTPException, status, Response # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import logging
from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv  # type: ignore
import os, requests, json, pickle

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
from qiskit_ibm_runtime import Estimator  # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import sys
from .config import load_api_keys
from qiskit.quantum_info import SparsePauliOp

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

def log_step(step: str, detail: str):
    logging.info(f"{step}: {detail}")

# Load API keys from environment variables or GitHub secrets
FMP_API_KEY, IBM_CLOUD_API_KEY, IBM_QUANTUM_API_TOKEN = load_api_keys()

# FastAPI app to expose logs via Swagger UI
app = FastAPI()

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://localhost:3000", "http://172.26.48.1:3000"]
    allow_credentials=True,
    allow_methods=["*"],
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
def predict_stock(current_user: dict[str, Any] = Depends(get_current_user)):
    log_step("User", f"Prediction requested by user: {current_user.get('username', 'unknown')}")
    log_step("Quantum Circuit", "Creating quantum circuit for stock prediction in qasm simulator")
    # Simple quantum circuit: 1 qubit, 1 classical bit
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Superposition: simulates uncertainty in stock movement
    qc.measure(0, 0)
    backend = Aer.get_backend('qasm_simulator')  # type: ignore
    job = backend.run(qc, shots=100)
    result = job.result()
    counts = result.get_counts(qc)
    log_step("Execution", f"Quantum circuit executed, counts: {counts}")
    # Interpret result: 0 = stock down, 1 = stock up
    prediction = "up" if counts.get('1', 0) > counts.get('0', 0) else "down"
    log_step("Prediction", f"Predicted stock movement: {prediction}")
    return {"prediction": prediction, "counts": counts}

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
    
# --- new code added for stock prediction ---

# --- Data Fetching ---
# --- Authentication Dependency ---
def check_ibm_keys():
    if not IBM_CLOUD_API_KEY:
        log_step("APIKeyCheck", "IBM_CLOUD_API_KEY is empty or not set.")
    if not IBM_QUANTUM_API_TOKEN:
        log_step("APIKeyCheck", "IBM_QUANTUM_API_TOKEN is empty or not set.")
    if not IBM_CLOUD_API_KEY or not IBM_QUANTUM_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid IBM Cloud or Quantum API Key")
    
    if not FMP_API_KEY or FMP_API_KEY == "FMP_API_KEY":
        log_step("APIKeyCheck", "FMP_API_KEY is empty, invalid, or not set.")
        raise HTTPException(status_code=401, detail="Invalid Financial Modeling Prep API Key")

def fetch_stock_data(symbol: str) -> pd.DataFrame:
    log_step("DataFetch", f"Fetching stock data for symbol: {symbol}")
    if not FMP_API_KEY or FMP_API_KEY == "FMP_API_KEY":
        log_step("DataFetch", "FMP_API_KEY is not set or is invalid.")
        #raise HTTPException(status_code=500, detail="FMP_API_KEY is not set or is invalid.")
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FMP_API_KEY}" #22CeNg7buOXdCS5Veour2wyJD7QnkOKV
    #url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey=22CeNg7buOXdCS5Veour2wyJD7QnkOKV"
    log_step("DataFetch", f"Requesting URL: {url}")
    r = requests.get(url)
    log_step("DataFetch", f"HTTP status: {r.status_code}")
    if r.status_code != 200:
        log_step("DataFetch", f"Failed to fetch data for {symbol}, status code: {r.status_code}, response: {r.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch data")
    data = r.json()
    log_step("DataFetch", f"Data fetched for {symbol}, normalizing JSON")
    if 'historical' not in data or not data['historical']:
        log_step("DataFetch", f"No historical data found in response for {symbol}. Response: {data}")
        raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
    df: pd.DataFrame = pd.json_normalize(data, 'historical', ['symbol'])
    df = df.sort_values('date').reset_index(drop=True)
    log_step("DataFetch", f"DataFrame created for {symbol}, shape: {df.shape}")
    return df

def get_today_str():
    return datetime.now().strftime('%Y-%m-%d')

def fetch_and_cache_stock_data(symbol: str) -> pd.DataFrame:
    today = get_today_str()
    cache_file = f"{symbol}_historical_{today}.csv"
    if os.path.exists(cache_file):
        log_step("DataFetch", f"Loading cached data for {symbol} from {cache_file}")
        return pd.read_csv(cache_file)
    df = fetch_stock_data(symbol)
    df.to_csv(cache_file, index=False)
    log_step("DataFetch", f"Fetched and cached data for {symbol} to {cache_file}")
    return df

# --- Model Save/Load ---
MODEL_PATH: str = "quantum1_model.pkl"
DATA_PATH: str = "quantum1_train_data.json"

def save_model(model: Any, path: str = MODEL_PATH) -> None:
    log_step("ModelSave", f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log_step("ModelSave", f"Model saved to {path}")

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

def save_train_data(data: dict[str, Any], path: str = DATA_PATH) -> None:
    log_step("DataSave", f"Saving training data to {path}")
    with open(path, "w") as f:
        json.dump(data, f)
    log_step("DataSave", f"Training data saved to {path}")

def load_train_data(path: str = DATA_PATH) -> dict[str, Any]:
    log_step("DataLoad", f"Loading training data from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    log_step("DataLoad", f"Training data loaded from {path}")
    return data

# --- Feature Engineering ---
from typing import Tuple, Any
from numpy.typing import NDArray

def make_features(
    df: pd.DataFrame,
    window: int = 2,
    col: str = 'open',
    n_points: int = 500
) -> Tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
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

# --- Quantum ML (Qiskit 1.0.0, IBM Quantum backend) ---
def build_ansatz(n: int, depth: int) -> QuantumCircuit:
    log_step("QuantumML", f"Building ansatz with n={n}, depth={depth}")
    qc = QuantumCircuit(n)
    for j in range(depth):
        for i in range(n):
            param_name = f'theta_{j}_{i}'
            theta_param = Parameter(param_name)
            qc.rx(theta_param, i)
            qc.ry(theta_param, i)
            qc.rz(theta_param, i)
    for i in range(n):
        if i == n - 1:
            qc.cx(i, 0)  # Use 'cx' for CNOT, not 'cnot'
        else:
            qc.cx(i, i + 1)
    log_step("QuantumML", "Ansatz built")
    return qc

def quantum_predict(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    x_test: np.ndarray[Any, Any],
    backend_name: str = "ibm_brisbane"
) -> Tuple[np.ndarray[Any, np.dtype[Any]], Any]:
    log_step("QuantumML", f"Starting quantum prediction on backend {backend_name}")
    num_features = x_train.shape[1]
    qreg = QuantumRegister(num_features, 'q')
    feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = QuantumCircuit(qreg)
    params = ParameterVector('theta', length=num_features)
    for i in range(num_features):
        ansatz.ry(params[i], qreg[i])
    optimizer = ADAM(maxiter=100)
    service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_QUANTUM_API_TOKEN)
    #backend = service.backend("ibm_brisbane")
    backend = service.backend(backend_name)
    log_step("QuantumML", f"Using backend: {backend.name()}")
    # Transpile feature map for backend compatibility
    from qiskit import transpile
    feature_map = transpile(feature_map, backend)
    # Setup VQR
    vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    log_step("QuantumML", "Fitting VQR model")
    vqr.fit(x_train, y_train)
    log_step("QuantumML", "Model fit complete, predicting")
    y_pred = vqr.predict(x_test)
    log_step("QuantumML", "Quantum prediction complete")
    return y_pred, vqr

# --- API Endpoints ---

from typing import List, Dict, Any

@app.get("/historical-data/{symbol}")
def api_historical_data(symbol: str, request: Request) -> List[Dict[str, Any]]:
    log_step("API", f"GET /historical-data/{symbol} called")
    check_ibm_keys()
    df = fetch_stock_data(symbol)
    log_step("API", f"Returning historical data for {symbol}")
    records = df.to_dict(orient="records")
    return [{str(k): v for k, v in record.items()} for record in records]

@app.post("/predict/classical")
def api_predict_classical(symbols: List[str], request: Request) -> Dict[str, Dict[str, float | List[Any] | str]]:
    log_step("API", f"POST /predict/classical called for symbols: {symbols}")
    check_ibm_keys()
    results: dict[str, dict[str, float | list[Any] | str]] = {}
    for symbol in symbols:
        log_step("API", f"Processing symbol: {symbol}")
        ann_model_path = f"{symbol}_classical_ann.pkl"
        ann_data_path = f"{symbol}_ann_train_data.json"
        if os.path.exists(ann_model_path) and os.path.exists(ann_data_path):
            log_step("API", f"Using ANN model/data for {symbol}")
            model = load_model(ann_model_path)
            data: dict[str, list[float]] = load_train_data(ann_data_path)
            x = np.array(data["x_train"], dtype=np.float64)
            y = np.array(data["y_train"], dtype=np.float64)
            y_pred = model.predict(x)
            mse = mean_squared_error(y, y_pred)
            results[symbol] = {
                "dates": [],
                "y_test": y.tolist(),
                "y_pred": y_pred.tolist(),
                "mse": mse,
                "model_type": "ann"
            }
        else:
            log_step("API", f"ANN model/data not found for {symbol}, using default classical model.")
            df = fetch_stock_data(symbol)
            x, y, dates = make_features(df)
            x_train = x[:400]
            x_test = x[400:]
            y_train = y[:400]
            y_test = y[400:]
            y_pred, model = classical_predict(x_train, y_train, x_test)
            mse = mean_squared_error(y_test, y_pred)
            results[symbol] = {
                "dates": dates[400:],
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "mse": mse,
                "model_type": "linear_regression"
            }
            save_model(model, f"{symbol}_classical.pkl")
            save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_train_data.json")
        log_step("API", f"Classical prediction complete for {symbol}")
    log_step("API", "Returning classical prediction results")
    return results

@app.post("/predict/quantum")
def api_predict_quantum(symbols: list[str], request: Request) -> dict[str, dict[str, float | list | str]]:
    log_step("API", f"POST /predict/quantum called for symbols: {symbols}")
    check_ibm_keys()
    results: dict[str, dict[str, float | list[Any] | str]] = {}
    for symbol in symbols:
        log_step("API", f"Processing symbol: {symbol}")
        qnn_model_path = f"{symbol}_quantum_qnn.pkl"
        qnn_data_path = f"{symbol}_qnn_train_data.json"
        if os.path.exists(qnn_model_path) and os.path.exists(qnn_data_path):
            log_step("API", f"Using QNN model/data for {symbol}")
            model = load_model(qnn_model_path)
            data: dict[str, list[float]] = load_train_data(qnn_data_path)
            x = np.array(data["x_train"], dtype=np.float64)
            y = np.array(data["y_train"], dtype=np.float64)
            y_pred = model.predict(x)
            mse = mean_squared_error(y, y_pred)
            results[symbol] = {
                "dates": [],
                "y_test": y.tolist(),
                "y_pred": y_pred.tolist(),
                "mse": mse,
                "model_type": "qnn"
            }
        else:
            log_step("API", f"QNN model/data not found for {symbol}, using default quantum model.")
            df = fetch_stock_data(symbol)
            x, y, dates = make_features(df)
            x_train = x[:400]
            x_test = x[400:]
            y_train = y[:400]
            y_test = y[400:]
            y_pred, vqr = quantum_predict(x_train, y_train, x_test)
            mse = mean_squared_error(y_test, y_pred)
            results[symbol] = {
                "dates": dates[400:],
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "mse": mse,
                "model_type": "vqr"
            }
            save_model(vqr, f"{symbol}_quantum.pkl")
            save_train_data({"x_train": x_train.tolist(), "y_train": y_train.tolist()}, f"{symbol}_train_data.json")
        log_step("API", f"Quantum prediction complete for {symbol}")
    log_step("API", "Returning quantum prediction results")
    return results

@app.get("/model/data/{symbol}/{model_type}")
def api_model_data(symbol: str, model_type: str, request: Request) -> dict[str, list[float]]:
    log_step("API", f"GET /model/data/{symbol}/{model_type} called")
    check_ibm_keys()
    if model_type == "ann":
        data: dict[str, list[float]] = load_train_data(f"{symbol}_ann_train_data.json")
    elif model_type == "qnn":
        data = load_train_data(f"{symbol}_qnn_train_data.json")
    elif model_type == "classical":
        data = load_train_data(f"{symbol}_train_data.json")
    elif model_type == "quantum":
        data = load_train_data(f"{symbol}_train_data.json")
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")
    log_step("API", f"Returning model data for {symbol} ({model_type})")
    return data

@app.get("/model/load/{symbol}/{model_type}")
def api_model_load(symbol: str, model_type: str, request: Request) -> dict[str, list[float]]:
    log_step("API", f"GET /model/load/{symbol}/{model_type} called")
    check_ibm_keys()
    if model_type == "ann":
        model = load_model(f"{symbol}_classical_ann.pkl")
        data: dict[str, list[float]] = load_train_data(f"{symbol}_ann_train_data.json")
    elif model_type == "qnn":
        model = load_model(f"{symbol}_quantum_qnn.pkl")
        data = load_train_data(f"{symbol}_qnn_train_data.json")
    elif model_type == "classical":
        model = load_model(f"{symbol}_classical.pkl")
        data = load_train_data(f"{symbol}_train_data.json")
    elif model_type == "quantum":
        model = load_model(f"{symbol}_quantum.pkl")
        data = load_train_data(f"{symbol}_train_data.json")
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")
    x_train = np.array(data["x_train"], dtype=np.float64)
    y_train = np.array(data["y_train"], dtype=np.float64)
    y_pred = model.predict(x_train)
    log_step("API", f"Returning loaded model predictions for {symbol} ({model_type})")
    return {"y_pred": y_pred.tolist(), "y_train": y_train.tolist()}

# --- Top 10 stock symbols (example, can be customized) ---
TOP_10_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH"
]

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

# --- Quantum QNN Training ---
from qiskit.primitives import Estimator # type: ignore
from qiskit.circuit.library import PauliFeatureMap # type: ignore
from scipy.optimize import minimize  # type: ignore

def train_quantum_qnn(symbols: list[str] = TOP_10_SYMBOLS) -> dict[str, str]:
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
            qreg = QuantumRegister(num_features, 'q')
            feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
            # Ansatz: simple Ry circuit
            ansatz = QuantumCircuit(qreg)
            params = ParameterVector('theta', length=num_features)
            for i in range(num_features):
                ansatz.ry(params[i], qreg[i])
            # Setup Estimator primitive
            from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
            service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_QUANTUM_API_TOKEN)
            backend = service.backend("ibm_brisbane")
            # Transpile the parameterized feature map once for backend compatibility
            from qiskit import transpile
            feature_map = transpile(feature_map, backend)
            with Session(service=service, backend=backend) as session:
                estimator = Estimator(session=session)
                # Objective function for classical optimizer
                def objective(theta):
                    values = []
                    for xi in x:
                        qc = QuantumCircuit(num_features)
                        # Feature map: assign parameters and append instructions one by one
                        feature_circ = feature_map.assign_parameters(xi)
                        for instr, qargs, cargs in feature_circ.data:
                            qc.append(instr, qargs, cargs)
                        # Ansatz
                        ansatz_circ = ansatz.assign_parameters(theta)
                        for instr, qargs, cargs in ansatz_circ.data:
                            qc.append(instr, qargs, cargs)
                        # Z observable on first qubit
                        observable = SparsePauliOp("Z" + "I" * (num_features - 1))
                        value = estimator.run(qc, observable).result().values[0]
                        values.append(value)
                    return np.mean((np.array(values) - y) ** 2)
                theta0 = np.random.rand(num_features)
                res = minimize(objective, theta0, method='COBYLA')
                # Save trained parameters
                save_model(res.x, f"{symbol}_quantum_qnn_params.pkl")
                save_train_data({"x_train": x.tolist(), "y_train": y.tolist()}, f"{symbol}_qnn_train_data.json")
                log_step("TrainQuantumQNN", f"Trained and saved QNN params for {symbol}")
                results[symbol] = "trained"
        except HTTPException as e:
            log_step("TrainQuantumQNN", f"HTTPException for {symbol}: {e.detail}")
            results[symbol] = f"error: {e.detail}"
        except Exception as e:
            log_step("TrainQuantumQNN", f"Exception for {symbol}: {str(e)}")
            results[symbol] = f"error: {str(e)}"
    return results

# --- FastAPI Endpoints for Training ---
@app.post("/train/classical")
def api_train_classical(_request: Request) -> dict[str, object]:
    check_ibm_keys()
    log_step("API", "/train/classical called")
    result = train_classical_ann()
    log_step("API", "Classical ANN training complete")
    return {"status": "success", "trained": result}

@app.post("/train/quantum")
def api_train_quantum(_request: Request) -> dict[str, object]:
    check_ibm_keys()
    log_step("API", "/train/quantum called")
    result = train_quantum_qnn()
    log_step("API", "Quantum QNN training complete")
    return {"status": "success", "trained": result}

# --- Helper: Check and auto-train if model missing in prediction endpoints ---
def ensure_classical_ann(symbol: str):
    model_path = f"{symbol}_classical_ann.pkl"
    if not os.path.exists(model_path):
        log_step("AutoTrain", f"Classical ANN model missing for {symbol}, training now.")
        train_classical_ann([symbol])

def ensure_quantum_qnn(symbol: str):
    model_path = f"{symbol}_quantum_qnn.pkl"
    if not os.path.exists(model_path):
        log_step("AutoTrain", f"Quantum QNN model missing for {symbol}, training now.")
        train_quantum_qnn([symbol])

from scipy.optimize import minimize

def objective(theta, x, y):
    # Build circuit for each input
    values = []
    for xi in x:
        qc = QuantumCircuit(num_features)
        # Encode features
        feature_circ = feature_map.assign_parameters(xi)
        qc.compose(feature_circ, inplace=True)
        # Add ansatz
        ansatz_circ = ansatz.assign_parameters(theta)
        qc.compose(ansatz_circ, inplace=True)
        # Measure expectation value of Z on first qubit
        observable = SparsePauliOp("Z" + "I" * (num_features - 1))
        value = estimator.run(qc, observable).result().values[0]
        values.append(value)
    # Mean squared error
    return np.mean((np.array(values) - y) ** 2)

# The following lines are removed because num_features, x_train, and y_train are not defined in this scope,
# and similar optimization logic is already implemented inside the train_quantum_qnn function.

def predict(theta, x):
    preds = []
    for xi in x:
        qc = QuantumCircuit(num_features)
        feature_circ = feature_map.assign_parameters(xi)
        qc.compose(feature_circ, inplace=True)
        ansatz_circ = ansatz.assign_parameters(theta)
        qc.compose(ansatz_circ, inplace=True)
        observable = SparsePauliOp("Z" + "I" * (num_features - 1))
        value = estimator.run(qc, observable).result().values[0]
        preds.append(value)
    return np.array(preds)
# --- End of Code ---