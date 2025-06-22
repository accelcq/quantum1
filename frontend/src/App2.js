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
  const [predictQuantumSimSymbols, setPredictQuantumSimSymbols] = useState("");
  const [predictQuantumSimResult, setPredictQuantumSimResult] = useState(null);
  const [predictQuantumMachineSymbols, setPredictQuantumMachineSymbols] = useState("");
  const [predictQuantumMachineBackend, setPredictQuantumMachineBackend] = useState("ibm_brisbane");
  const [predictQuantumMachineResult, setPredictQuantumMachineResult] = useState(null);
  const [predictCompareSymbols, setPredictCompareSymbols] = useState("");
  const [predictCompareBackend, setPredictCompareBackend] = useState("ibm_brisbane");
  const [predictCompareResult, setPredictCompareResult] = useState(null);

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

      {/* POST /predict/quantum/simulator */}
      <section>
        <h2>Predict Quantum Simulator (/predict/quantum/simulator)</h2>
        <input placeholder="Comma-separated symbols" value={predictQuantumSimSymbols} onChange={e => setPredictQuantumSimSymbols(e.target.value)} />
        <button onClick={() => post("/predict/quantum/simulator", { symbols: predictQuantumSimSymbols.split(",").map(s => s.trim()) }, setPredictQuantumSimResult)}>Predict</button>
        <pre>{JSON.stringify(predictQuantumSimResult, null, 2)}</pre>
      </section>

      {/* POST /predict/quantum/machine/{backend} */}
      <section>
        <h2>Predict Quantum Machine (/predict/quantum/machine/&#123;backend&#125;)</h2>
        <input placeholder="Comma-separated symbols" value={predictQuantumMachineSymbols} onChange={e => setPredictQuantumMachineSymbols(e.target.value)} />
        <input placeholder="Backend (e.g. ibm_brisbane)" value={predictQuantumMachineBackend} onChange={e => setPredictQuantumMachineBackend(e.target.value)} />
        <button onClick={() => post(`/predict/quantum/machine/${predictQuantumMachineBackend}`, { symbols: predictQuantumMachineSymbols.split(",").map(s => s.trim()) }, setPredictQuantumMachineResult)}>Predict</button>
        <pre>{JSON.stringify(predictQuantumMachineResult, null, 2)}</pre>
      </section>

      {/* POST /predict/compare */}
      <section>
        <h2>Predict Compare (/predict/compare)</h2>
        <input placeholder="Comma-separated symbols" value={predictCompareSymbols} onChange={e => setPredictCompareSymbols(e.target.value)} />
        <input placeholder="Backend (e.g. ibm_brisbane)" value={predictCompareBackend} onChange={e => setPredictCompareBackend(e.target.value)} />
        <button onClick={() => post("/predict/compare", { symbols: predictCompareSymbols.split(",").map(s => s.trim()), backend: predictCompareBackend }, setPredictCompareResult)}>Predict</button>
        <pre>{JSON.stringify(predictCompareResult, null, 2)}</pre>
      </section>
    </div>
  );
}

export default App;