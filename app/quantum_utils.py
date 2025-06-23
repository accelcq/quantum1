import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.optimizers import ADAM
from qiskit_machine_learning.algorithms import VQR

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

def quantum_predict_simulator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray
):
    num_features = x_train.shape[1]
    feature_map = PauliFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = QuantumCircuit(num_features)
    params = ParameterVector('theta', length=num_features)
    for i in range(num_features):
        ansatz.ry(params[i], i)
    optimizer = ADAM(maxiter=100)

    if Aer is None:
        raise ImportError("Qiskit Aer is not installed.")
    # VQR uses the default local simulator if no backend is provided
    vqr = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)

    vqr.fit(x_train, y_train)
    y_pred = vqr.predict(x_test)
    return y_pred, vqr
