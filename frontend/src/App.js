import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000"; // Change if deploying

function App() {
  const [token, setToken] = useState("");
  const [username, setUsername] = useState("accelcq");
  const [password, setPassword] = useState("password123");
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);

  const login = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API_URL}/token`, new URLSearchParams({
        username, password
      }));
      console.log("Login response:", res.data); // Add this line
      setToken(res.data.access_token);
    } catch (err) {
      alert("Login failed");
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await axios.get(`${API_URL}/logs`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLogs(res.data);
    } catch {
      alert("Failed to fetch logs");
    }
  };

  const predictStock = async () => {
    try {
      const res = await axios.post(`${API_URL}/predict-stock`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(res.data.result);
    } catch {
      alert("Failed to run circuit");
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto" }}>
      <h1>AccelCQ Quantum API Demo</h1>
      {!token ? (
        <form onSubmit={login}>
          <h2>Login</h2>
          <input value={username} onChange={e => setUsername(e.target.value)} placeholder="Username" /><br />
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" /><br />
          <button type="submit">Login</button>
        </form>
      ) : (
        <>
          <button onClick={fetchLogs}>Fetch Logs</button>
          <button onClick={predictStock}>Predict Stock</button>
          <button onClick={() => setToken("")}>Logout</button>
          <h3>Logs</h3>
          <ul>
            {logs.map((log, i) => (
              <li key={i}><b>{log.timestamp}</b>: {log.message}</li>
            ))}
          </ul>
          {result && (
            <div>
              <h3>Result</h3>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;