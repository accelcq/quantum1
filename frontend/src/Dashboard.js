import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const API_URL = "http://localhost:8000";

const symbolsList = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX", "IBM", "INTC"];

const Dashboard = () => {
  const [token, setToken] = useState("");
  const [login, setLogin] = useState({ username: "", password: "" });
  const [symbol, setSymbol] = useState("AAPL");
  const [chartData, setChartData] = useState([]);
  const [responses, setResponses] = useState({});

  const authHeader = token ? { Authorization: `Bearer ${token}` } : {};

  const handleLogin = async () => {
    const formData = new URLSearchParams();
    formData.append("username", login.username);
    formData.append("password", login.password);
    const res = await fetch(`${API_URL}/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData
    });
    const data = await res.json();
    setToken(data.access_token);
  };

  const callAPI = async (method, url, body) => {
    const config = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...authHeader
      },
    };
    if (body) config.body = JSON.stringify(body);
    const res = await fetch(`${API_URL}${url}`, config);
    const data = await res.json();
    setResponses(prev => ({ ...prev, [url]: data }));
    return data;
  };

  const fetchChartData = async () => {
    const res = await fetch(`${API_URL}/historical-data/${symbol}`);
    const data = await res.json();
    setChartData(data.map(day => ({ date: day.date, close: day.close })));
    setResponses(prev => ({ ...prev, ["/historical"]: data }));
  };

  const exportCSV = (data, filename) => {
    const csv = [Object.keys(data[0]).join(","), ...data.map(row => Object.values(row).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  };

  return (
    <div className="p-4 md:p-10 bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen text-white">
      <div className="flex items-center justify-between mb-6 flex-col md:flex-row gap-4">
        <img src="/accelcq-logo.svg" alt="AccelCQ Logo" width={160} height={60} />
        <h1 className="text-3xl font-bold text-center">Quantum1 API Dashboard</h1>
      </div>

      {/* Login */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">1. Login</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Username" onChange={e => setLogin({ ...login, username: e.target.value })} />
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" type="password" placeholder="Password" onChange={e => setLogin({ ...login, password: e.target.value })} />
        <button onClick={handleLogin} className="w-full bg-green-600 py-2 rounded-lg">Login</button>
        <p className="mt-2 text-sm break-all">Token: {token ? `${token.slice(0, 20)}...` : "Not logged in"}</p>
      </motion.div>

      {/* Historical Data */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">2. Historical Data</h2>
        <select className="w-full px-4 py-2 mb-2 rounded-lg text-black" value={symbol} onChange={e => setSymbol(e.target.value)}>
          {symbolsList.map(sym => <option key={sym} value={sym}>{sym}</option>)}
        </select>
        <button onClick={fetchChartData} className="w-full bg-blue-600 py-2 rounded-lg">Fetch</button>
        <ResponsiveContainer width="100%" height={300} className="mt-4">
          <LineChart data={chartData}>
            <XAxis dataKey="date" hide />
            <YAxis domain={['dataMin', 'dataMax']} />
            <Tooltip />
            <CartesianGrid strokeDasharray="3 3" />
            <Line type="monotone" dataKey="close" stroke="#4ade80" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        {chartData.length > 0 && (
          <button onClick={() => exportCSV(chartData, `${symbol}_historical.csv`)} className="mt-2 bg-yellow-500 py-1 px-3 rounded-lg">Export CSV</button>
        )}
      </motion.div>

      {/* Classical Train */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">3. Train Classical Model</h2>
        <button onClick={() => callAPI("POST", "/train/classical", [symbol])} className="w-full bg-indigo-600 py-2 rounded-lg">Train</button>
        <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/train/classical"], null, 2)}</pre>
      </motion.div>

      {/* Quantum Train */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">4. Train Quantum Model</h2>
        <button onClick={() => callAPI("POST", "/train/quantum", [symbol])} className="w-full bg-indigo-600 py-2 rounded-lg">Train</button>
        <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/train/quantum"], null, 2)}</pre>
      </motion.div>

      {/* Predict QNN */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">5. Predict using QNN</h2>
        <button onClick={() => callAPI("POST", "/predict/quantum", [symbol])} className="w-full bg-teal-600 py-2 rounded-lg">Predict</button>
        {responses["/predict/quantum"] && (
          <>
            <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/predict/quantum"], null, 2)}</pre>
            <button onClick={() => exportCSV(responses["/predict/quantum"], `${symbol}_qnn_prediction.csv`)} className="mt-2 bg-yellow-500 py-1 px-3 rounded-lg">Export CSV</button>
          </>
        )}
      </motion.div>

      {/* Predict ANN */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">6. Predict using ANN</h2>
        <button onClick={() => callAPI("POST", "/predict/classical", [symbol])} className="w-full bg-teal-600 py-2 rounded-lg">Predict</button>
        {responses["/predict/classical"] && (
          <>
            <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/predict/classical"], null, 2)}</pre>
            <button onClick={() => exportCSV(responses["/predict/classical"], `${symbol}_ann_prediction.csv`)} className="mt-2 bg-yellow-500 py-1 px-3 rounded-lg">Export CSV</button>
          </>
        )}
      </motion.div>
    </div>
  );
};

export default Dashboard;
