# Quantum utilities with fallback prediction - Updated for new IBM Quantum token (July 2025)
import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from typing import Any, Tuple

# Import logging
import logging

# Import Aer and VQR with compatibility for Qiskit 1.4.1+
try:
    from qiskit_aer import Aer
    from qiskit_aer.primitives import Sampler, Estimator
except ImportError:
    Aer = None
    Sampler = None
    Estimator = None

try:
    from qiskit_machine_learning.algorithms import VQR
except ImportError:
    VQR = None

def predict_stock_quantum_simulator(symbol: str, days: int) -> Tuple[np.ndarray, Any]:
    """Quantum simulator prediction with Qiskit 1.4.1 compatibility and fallback"""
    try:
        stock_data = fetch_stock_data(symbol)
        if stock_data.empty:
            raise ValueError(f"No stock data available for {symbol}")

        # Fallback if VQR or Aer is not available
        if VQR is None or Aer is None:
            logging.warning("VQR or Aer not available, using fallback prediction.")
            return quantum_inspired_fallback_prediction(stock_data, days)

        # Prepare features
        X, _ = make_features(stock_data)
        if len(X) < 10:
            logging.warning("Insufficient data for VQR, using fallback.")
            return quantum_inspired_fallback_prediction(stock_data, days)

        # Use Aer simulator
        backend = Aer.get_backend('aer_simulator')

        # Create quantum feature map
        feature_map = PauliFeatureMap(feature_dimension=min(len(X[0]), 4), reps=1)

        # Create VQR
        vqr = VQR(
            feature_map=feature_map,
            ansatz=feature_map,
            optimizer=None,  # Use default optimizer or set as needed
            quantum_instance=backend
        )

        y = stock_data['close'].values[-len(X):]
        vqr.fit(X, y)

        # Predict
        last_features = X[-1].reshape(1, -1)
        predictions = []
        for _ in range(days):
            pred = vqr.predict(last_features)[0]
            predictions.append(pred)
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred

        return np.array(predictions), {"method": "VQR", "backend": "aer_simulator"}

    except Exception as e:
        logging.error(f"Quantum prediction error: {str(e)}. Using fallback.")
        return quantum_inspired_fallback_prediction(stock_data, days)

def quantum_inspired_fallback_prediction(stock_data, days):
    """Quantum-inspired fallback when VQR fails"""
    prices = stock_data['close'].values
    trend = np.mean(np.diff(prices[-10:]))
    volatility = np.std(prices[-10:])
    predictions = []
    last_price = prices[-1]
    for i in range(days):
        quantum_noise = np.random.normal(0, volatility * 0.1)
        prediction = last_price + trend * (i + 1) + quantum_noise
        predictions.append(prediction)
    return np.array(predictions), {"method": "quantum_inspired", "backend": "classical_simulation"}
