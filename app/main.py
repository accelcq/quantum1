from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import logging
from typing import Any, List
from datetime import datetime, timezone

# Qiskit 1.0.0 imports for quantum stock prediction
from qiskit import QuantumCircuit # type: ignore
from qiskit_aer import Aer  # type: ignore

# Set up logging to a file (simulate IBM Cloud Object Storage)
log_filename = "execution_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

def log_step(step: str, detail: str):
    logging.info(f"{step}: {detail}")

# FastAPI app to expose logs via Swagger UI
app = FastAPI()

# JWT Config
SECRET_KEY = "your-secret-key"  # Replace with a secure key in production!
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

from typing import Optional, Dict, Any

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict[str, Any]) -> str:
    from datetime import timedelta, datetime, timezone
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

@app.post("/predict-stock")
def predict_stock(current_user: dict[str, Any] = Depends(get_current_user)):
    log_step("User", f"Prediction requested by user: {current_user.get('username', 'unknown')}")
    log_step("Quantum Circuit", "Creating quantum circuit for stock prediction")
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

# To run: uvicorn main:app --reload
