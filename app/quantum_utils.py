import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.optimizers import ADAM
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import VQR
from .main import log_step

def quantum_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    backend_name: str = "ibm_brisbane"
):
    log_step("QuantumML", f"Starting quantum prediction on backend {backend_name}")
    num_features = x_train.shape[1]
    feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = QuantumCircuit(num_features)
    params = ParameterVector('theta', length=num_features)
    for i in range(num_features):
        ansatz.ry(params[i], i)
    optimizer = ADAM(maxiter=100)
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(backend_name)
    log_step("QuantumML", f"Using backend: {backend}")
    vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    log_step("QuantumML", "Fitting VQR model")
    vqr.fit(x_train, y_train)
    log_step("QuantumML", "Model fit complete, predicting")
    y_pred = vqr.predict(x_test)
    log_step("QuantumML", "Quantum prediction complete")
    return y_pred, vqr
