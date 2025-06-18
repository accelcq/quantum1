import React, { useState } from "react";

const API_URL = "http://localhost:8000"; // Change if needed

function App() {
  // State for each API input/output
  const [loginData, setLoginData] = useState({ username: "", password: "" });
  const [token, setToken] = useState(null); // token is the whole object from /token
  const [predictSimSymbol, setPredictSimSymbol] = useState("");
  const [predictSimResult, setPredictSimResult] = useState(null);
  const [logs, setLogs] = useState("");
  const [historicalSymbol, setHistoricalSymbol] = useState("");
  const [historicalData, setHistoricalData] = useState(null);
  const [predictClassicalSymbols, setPredictClassicalSymbols] = useState("");
  const [predictClassicalResult, setPredictClassicalResult] = useState(null);
  const [predictQuantumSymbols, setPredictQuantumSymbols] = useState("");
  const [predictQuantumResult, setPredictQuantumResult] = useState(null);
  const [modelDataSymbol, setModelDataSymbol] = useState("");
  const [modelDataType, setModelDataType] = useState("");
  const [modelData, setModelData] = useState(null);
  const [modelLoadSymbol, setModelLoadSymbol] = useState("");
  const [modelLoadType, setModelLoadType] = useState("");
  const [modelLoad, setModelLoad] = useState(null);
  const [trainClassicalSymbols, setTrainClassicalSymbols] = useState("");
  const [trainClassicalResult, setTrainClassicalResult] = useState(null);
  const [trainQuantumSymbols, setTrainQuantumSymbols] = useState("");
  const [trainQuantumResult, setTrainQuantumResult] = useState(null);

  const getAuthHeader = () =>
    token && token.access_token
      ? { Authorization: `Bearer ${token.access_token}` }
      : {};

  // Helper for POST requests
  const post = async (endpoint, body, setter) => {
    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify(body),
      });
      setter(await res.json());
    } catch (e) {
      setter({ error: e.message });
    }
  };

  // Helper for GET requests
  const get = async (endpoint, setter) => {
    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        headers: {
          ...getAuthHeader(),
        },
      });
      setter(await res.json());
    } catch (e) {
      setter({ error: e.message });
    }
  };

  const login = async () => {
    const params = new URLSearchParams();
    params.append("username", loginData.username);
    params.append("password", loginData.password);

    try {
      const res = await fetch(`${API_URL}/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params,
      });
      setToken(await res.json());
    } catch (e) {
      setToken({ error: e.message });
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Quantum1 API Dashboard</h1>

      {/* POST /token */}
      <section>
        <h2>Login (/token)</h2>
        <input placeholder="Username" value={loginData.username} onChange={e => setLoginData({ ...loginData, username: e.target.value })} />
        <input placeholder="Password" type="password" value={loginData.password} onChange={e => setLoginData({ ...loginData, password: e.target.value })} />
        <button onClick={login}>Login</button>
        <pre>{JSON.stringify(token, null, 2)}</pre>
      </section>

      {/* POST /predict-stock-simulator */}
      <section>
        <h2>Predict Stock Simulator (/predict-stock-simulator)</h2>
        <input placeholder="Stock Symbol" value={predictSimSymbol} onChange={e => setPredictSimSymbol(e.target.value)} />
        <button onClick={() => post("/predict-stock-simulator", { symbol: predictSimSymbol }, setPredictSimResult)}>Predict</button>
        <pre>{JSON.stringify(predictSimResult, null, 2)}</pre>
      </section>

      {/* GET /logs */}
      <section>
        <h2>Get Logs (/logs)</h2>
        <button onClick={() => get("/logs", setLogs)}>Get Logs</button>
        <pre>{JSON.stringify(logs, null, 2)}</pre>
      </section>

      {/* GET / */}
      <section>
        <h2>Root (/)</h2>
        <button onClick={() => get("/", setLogs)}>Call Root</button>
        <pre>{JSON.stringify(logs, null, 2)}</pre>
      </section>

      {/* GET /health */}
      <section>
        <h2>Health Check (/health)</h2>
        <button onClick={() => get("/health", setLogs)}>Check Health</button>
        <pre>{JSON.stringify(logs, null, 2)}</pre>
      </section>

      {/* GET /version */}
      <section>
        <h2>Version (/version)</h2>
        <button onClick={() => get("/version", setLogs)}>Get Version</button>
        <pre>{JSON.stringify(logs, null, 2)}</pre>
      </section>

      {/* GET /historical-data/{symbol} */}
      <section>
        <h2>Api Historical Data (/historical-data/&lt;symbol&gt;)</h2>
        <input placeholder="Stock Symbol" value={historicalSymbol} onChange={e => setHistoricalSymbol(e.target.value)} />
        <button onClick={() => get(`/historical-data/${historicalSymbol}`, setHistoricalData)}>Get Data</button>
        <pre>{JSON.stringify(historicalData, null, 2)}</pre>
      </section>

      {/* POST /predict/classical */}
      <section>
        <h2>Api Predict Classical (/predict/classical)</h2>
        <input placeholder="Comma-separated symbols" value={predictClassicalSymbols} onChange={e => setPredictClassicalSymbols(e.target.value)} />
        <button onClick={() => post("/predict/classical", predictClassicalSymbols.split(",").map(s => s.trim()), setPredictClassicalResult)}>Predict</button>
        <pre>{JSON.stringify(predictClassicalResult, null, 2)}</pre>
      </section>

      {/* POST /predict/quantum */}
      <section>
        <h2>Api Predict Quantum (/predict/quantum)</h2>
        <input placeholder="Comma-separated symbols" value={predictQuantumSymbols} onChange={e => setPredictQuantumSymbols(e.target.value)} />
        <button onClick={() => post("/predict/quantum", predictQuantumSymbols.split(",").map(s => s.trim()), setPredictQuantumResult)}>Predict</button>
        <pre>{JSON.stringify(predictQuantumResult, null, 2)}</pre>
      </section>

      {/* GET /model/data/{symbol}/{model_type} */}
      <section>
        <h2>Api Model Data (/model/data/&lt;symbol&gt;/&lt;model_type&gt;)</h2>
        <input placeholder="Symbol" value={modelDataSymbol} onChange={e => setModelDataSymbol(e.target.value)} />
        <input placeholder="Model Type (ann/qnn/classical/quantum)" value={modelDataType} onChange={e => setModelDataType(e.target.value)} />
        <button onClick={() => get(`/model/data/${modelDataSymbol}/${modelDataType}`, setModelData)}>Get Model Data</button>
        <pre>{JSON.stringify(modelData, null, 2)}</pre>
      </section>

      {/* GET /model/load/{symbol}/{model_type} */}
      <section>
        <h2>Api Model Load (/model/load/&lt;symbol&gt;/&lt;model_type&gt;)</h2>
        <input placeholder="Symbol" value={modelLoadSymbol} onChange={e => setModelLoadSymbol(e.target.value)} />
        <input placeholder="Model Type (ann/qnn/classical/quantum)" value={modelLoadType} onChange={e => setModelLoadType(e.target.value)} />
        <button onClick={() => get(`/model/load/${modelLoadSymbol}/${modelLoadType}`, setModelLoad)}>Load Model</button>
        <pre>{JSON.stringify(modelLoad, null, 2)}</pre>
      </section>

      {/* POST /train/classical */}
      <section>
        <h2>Api Train Classical (/train/classical)</h2>
        <input placeholder="Comma-separated symbols" value={trainClassicalSymbols} onChange={e => setTrainClassicalSymbols(e.target.value)} />
        <button onClick={() => post("/train/classical", trainClassicalSymbols.split(",").map(s => s.trim()), setTrainClassicalResult)}>Train</button>
        <pre>{JSON.stringify(trainClassicalResult, null, 2)}</pre>
      </section>

      {/* POST /train/quantum */}
      <section>
        <h2>Api Train Quantum (/train/quantum)</h2>
        <input placeholder="Comma-separated symbols" value={trainQuantumSymbols} onChange={e => setTrainQuantumSymbols(e.target.value)} />
        <button onClick={() => post("/train/quantum", trainQuantumSymbols.split(",").map(s => s.trim()), setTrainQuantumResult)}>Train</button>
        <pre>{JSON.stringify(trainQuantumResult, null, 2)}</pre>
      </section>
    </div>
  );
}

export default App;