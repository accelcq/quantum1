import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.optimizers import ADAM
from qiskit_machine_learning.algorithms import VQR
from .main import log_step

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

    if backend_name == "aer_simulator":
        if Aer is None:
            raise ImportError("Qiskit Aer is not installed.")
        log_step("QuantumML", "Using local Aer simulator backend")
        print(f"Using Aer backend: {Aer.name()}")
        backend = Aer.get_backend(backend_name)
        # VQR uses the default local simulator if no backend is provided
        vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    else:
        if QiskitRuntimeService is None:
            raise ImportError("Qiskit IBM Runtime is not installed.")
        log_step("QuantumML", f"Using IBM Quantum backend: {backend_name}")
        print(f"Using IBM Quantum backend: {backend_name}")
        print(f"Using IBMQ_API_TOKEN: {IBMQ_API_TOKEN is not None}")

        service = QiskitRuntimeService(channel="ibm_quantum" token="IBMQ_API_TOKEN")
        backend = service.backend(backend_name)
        vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, quantum_instance=backend)

    log_step("QuantumML", "Fitting VQR model")
    vqr.fit(x_train, y_train)
    log_step("QuantumML", "Model fit complete, predicting")
    y_pred = vqr.predict(x_test)
    log_step("QuantumML", "Quantum prediction complete")
    return y_pred, vqr
