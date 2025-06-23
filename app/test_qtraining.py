import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_train_quantum_simulator_default():
    response = client.post("/train/quantum/simulator", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "AAPL" in data["trained"]

def test_train_quantum_simulator_multi():
    response = client.post("/train/quantum/simulator", json={"symbols": ["AAPL", "GOOGL"]})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "AAPL" in data["trained"]
    assert "GOOGL" in data["trained"]
