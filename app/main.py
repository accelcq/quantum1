from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
from qiskit import QuantumCircuit

# Initialize the service with your IBM Cloud account
service = QiskitRuntimeService()

# Access the ibmq_qasm_simulator backend
backend = service.backend("ibmq_qasm_simulator")

# Create a simple quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Execute the circuit using the Sampler primitive
from qiskit.primitives import Sampler
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid

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
fake_users_db = {
    "accelcq": {
        "username": "accelcq",
        "full_name": "AccelCQ User",
        "hashed_password": pwd_context.hash("password123"),
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict):
    from datetime import timedelta, datetime
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
        username: str = payload.get("sub")
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

@app.post("/run-circuit")
def run_circuit(current_user: dict = Depends(get_current_user)):
    log_step("Service Initialization", "QiskitRuntimeService initialized")
    log_step("Backend Access", f"Backend '{backend.name}' accessed")
    log_step("Circuit Creation", f"QuantumCircuit with {qc.num_qubits} qubits created")
    sampler = Sampler(backend=backend)
    log_step("Execution", "Circuit executed using Sampler primitive")
    job = sampler.run(qc)
    result = job.result()
    log_step("Result", f"Result: {result}")
    return {"result": str(result)}

@app.get("/logs", response_model=list[LogEntry])
def get_logs(current_user: dict = Depends(get_current_user)):
    entries = []
    with open(log_filename, "r") as f:
        for line in f:
            # Split only on the first space to separate timestamp and message
            ts_msg = line.strip().split(" ", 1)
            if len(ts_msg) == 2:
                ts, msg = ts_msg
                entries.append(LogEntry(timestamp=ts, message=msg))
    return entries

# To run: uvicorn main:app --reload
