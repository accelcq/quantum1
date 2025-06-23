import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8080";

const symbolsList = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX", "IBM", "INTC"];

// Example: Top 10 stock symbols and company names
const topStocks = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corporation" },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "AMZN", name: "Amazon.com, Inc." },
  { symbol: "META", name: "Meta Platforms, Inc." },
  { symbol: "TSLA", name: "Tesla, Inc." },
  { symbol: "NVDA", name: "NVIDIA Corporation" },
  { symbol: "JPM", name: "JPMorgan Chase & Co." },
  { symbol: "V", name: "Visa Inc." },
  { symbol: "UNH", name: "UnitedHealth Group Incorporated" }
];

const Dashboard = () => {
  const [token, setToken] = useState("");
  const [login, setLogin] = useState({ username: "", password: "" });
  const [symbol, setSymbol] = useState("AAPL");
  const [chartData, setChartData] = useState([]);
  const [responses, setResponses] = useState({});
  const [backend, setBackend] = useState("ibm_brisbane");

  const authHeader = token ? { Authorization: `Bearer ${token}` } : {};

  const handleLogin = async () => {
    const data = new URLSearchParams();
    data.append('username', login.username);
    data.append('password', login.password);

    fetch(`${API_URL}/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: data,
    })
    .then(response => response.json())
    .then(data => {
      // handle success
      setToken(data.access_token);
    })
    .catch(error => {
      // handle error
    });
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

  const exportCSV = (data, filename = "export.csv") => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      alert("No data to export.");
      return;
    }
    const keys = Object.keys(data[0]);
    const csvRows = [
      keys.join(","), // header
      ...data.map(row => keys.map(k => JSON.stringify(row[k] ?? "")).join(","))
    ];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
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

      {/* Top 10 Stock Symbols */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Top 10 Stock Symbols</h2>
        <ul className="list-disc list-inside">
          {topStocks.map(stock => (
            <li key={stock.symbol}>
              <b>{stock.symbol}</b>: {stock.name}
            </li>
          ))}
        </ul>
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
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <button onClick={() => callAPI("POST", "/train/classicalML", { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-indigo-600 py-2 rounded-lg">Train using Classical ML</button>
        <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/train/classicalML"], null, 2)}</pre>
      </motion.div>

      {/* Quantum Train */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">4. Train Quantum Model (Simulator)</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <button onClick={() => callAPI("POST", "/train/quantum/simulator", { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-indigo-600 py-2 rounded-lg">Train using Quantum ML (Simulator)</button>
        <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/train/quantum/simulator"], null, 2)}</pre>
      </motion.div>

      {/* Quantum Train (Real Machine) */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">5. Train Quantum Model (Real Machine)</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Quantum backend (e.g. ibm_brisbane)" value={backend} onChange={e => setBackend(e.target.value)} />
        <button onClick={() => callAPI("POST", `/train/quantum/machine/${backend}`, { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-indigo-700 py-2 rounded-lg">Train using Quantum ML (Real Machine)</button>
        <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses[`/train/quantum/machine/${backend}`], null, 2)}</pre>
      </motion.div>

      {/* Predict using Quantum Simulator */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">6. Predict using Quantum Simulator</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <button onClick={() => callAPI("POST", "/predict/quantum/simulator", { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-teal-600 py-2 rounded-lg">Predict (Simulator)</button>
        {responses["/predict/quantum/simulator"] && (
          <>
            <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/predict/quantum/simulator"], null, 2)}</pre>
            <button onClick={() => exportCSV(responses["/predict/quantum/simulator"], `${symbol}_qnn_sim_prediction.csv`)} className="mt-2 bg-yellow-500 py-1 px-3 rounded-lg">Export CSV</button>
            {/* Chart for Quantum Simulator Prediction */}
            {symbol.split(",").map(s => s.trim()).filter(s => s).map(sym => (
              responses["/predict/quantum/simulator"][sym] && (
                <ResponsiveContainer width="100%" height={300} key={sym}>
                  <LineChart
                    data={responses["/predict/quantum/simulator"][sym].y_test.map((y, i) => ({
                      date: responses["/predict/quantum/simulator"][sym].dates[i],
                      Actual: y,
                      Predicted: responses["/predict/quantum/simulator"][sym].y_pred[i]
                    }))}
                  >
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="Actual" stroke="#4ade80" dot={false} />
                    <Line type="monotone" dataKey="Predicted" stroke="#818cf8" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )
            ))}
          </>
        )}
      </motion.div>

      {/* Predict using Real Quantum Machine */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">7. Predict using Real Quantum Machine</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Quantum backend (e.g. ibm_brisbane)" value={backend} onChange={e => setBackend(e.target.value)} />
        <button onClick={() => callAPI("POST", `/predict/quantum/machine/${backend}`, { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-teal-700 py-2 rounded-lg">Predict (Real Quantum)</button>
        {responses[`/predict/quantum/machine/${backend}`] && (
          <>
            <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses[`/predict/quantum/machine/${backend}`], null, 2)}</pre>
            <button onClick={() => exportCSV(responses[`/predict/quantum/machine/${backend}`], `${symbol}_qnn_real_prediction.csv`)} className="mt-2 bg-yellow-500 py-1 px-3 rounded-lg">Export CSV</button>
            {/* Chart for Quantum Machine Prediction */}
            {symbol.split(",").map(s => s.trim()).filter(s => s).map(sym => (
              responses[`/predict/quantum/machine/${backend}`][sym] && (
                <ResponsiveContainer width="100%" height={300} key={sym}>
                  <LineChart
                    data={responses[`/predict/quantum/machine/${backend}`][sym].y_test.map((y, i) => ({
                      date: responses[`/predict/quantum/machine/${backend}`][sym].dates[i],
                      Actual: y,
                      Predicted: responses[`/predict/quantum/machine/${backend}`][sym].y_pred[i]
                    }))}
                  >
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="Actual" stroke="#4ade80" dot={false} />
                    <Line type="monotone" dataKey="Predicted" stroke="#f472b6" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )
            ))}
          </>
        )}
      </motion.div>

      {/* Predict using Classical ML */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">8. Predict using Classical ML</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <button onClick={() => callAPI("POST", "/predict/classicalML", { symbols: symbol ? symbol.split(",").map(s => s.trim()) : ["AAPL"] })} className="w-full bg-teal-600 py-2 rounded-lg">Predict (Classical ML)</button>
        {responses["/predict/classicalML"] && (() => {
          const syms = symbol ? symbol.split(",").map(s => s.trim()).filter(s => s) : ["AAPL"];
          // Build combined data for chart
          let chartData = [];
          syms.forEach(sym => {
            const pred = responses["/predict/classicalML"][sym];
            if (pred && pred.y_pred && pred.dates) {
              pred.y_pred.forEach((y, i) => {
                chartData.push({
                  date: pred.dates[i],
                  [`${sym}_Predicted`]: y,
                  [`${sym}_Actual`]: pred.y_test?.[i]
                });
              });
            }
          });
          // Merge by date
          const merged = {};
          chartData.forEach(row => {
            if (!merged[row.date]) merged[row.date] = { date: row.date };
            Object.assign(merged[row.date], row);
          });
          const mergedData = Object.values(merged);
          const colors = ["#4ade80", "#818cf8", "#f472b6", "#fbbf24", "#06b6d4", "#a21caf", "#f59e42", "#e11d48"];
          return (
            <>
              <pre className="mt-2 text-xs bg-black text-white p-2 rounded overflow-x-auto">{JSON.stringify(responses["/predict/classicalML"], null, 2)}</pre>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={mergedData}>
                  <XAxis dataKey="date" hide />
                  <YAxis />
                  <Tooltip />
                  <CartesianGrid strokeDasharray="3 3" />
                  {syms.map((sym, idx) => (
                    <Line key={sym+"_Predicted"} type="monotone" dataKey={`${sym}_Predicted`} stroke={colors[idx % colors.length]} dot={false} />
                  ))}
                  {syms.map((sym, idx) => (
                    <Line key={sym+"_Actual"} type="monotone" dataKey={`${sym}_Actual`} stroke="#fbbf24" dot={false} strokeDasharray="3 3" />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </>
          );
        })()}
      </motion.div>

      {/* Prediction Comparison */}
      <motion.div className="bg-slate-700 shadow-xl rounded-2xl p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">9. Prediction Comparison (Classical vs Quantum)</h2>
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Stock symbol(s), comma-separated (default: AAPL)" value={symbol} onChange={e => setSymbol(e.target.value)} />
        <input className="w-full px-4 py-2 mb-2 rounded-lg text-black" placeholder="Quantum backend (default: ibm_brisbane)" value={backend} onChange={e => setBackend(e.target.value)} />
        <button onClick={async () => {
          const syms = symbol ? symbol.split(",").map(s => s.trim()).filter(s => s) : ["AAPL"];
          const be = backend || "ibm_brisbane";
          const data = await callAPI("POST", "/predict/compare", { symbols: syms, backend: be });
          setResponses(prev => ({ ...prev, "/predict/compare": data }));
        }} className="w-full bg-pink-600 py-2 rounded-lg">Compare Predictions</button>
        {responses["/predict/compare"] && symbol.split(",").map(s => s.trim()).filter(s => s).map(sym => {
          // Build combined data for chart
          const classical = responses["/predict/compare"][sym]?.classical;
          const quantum_sim = responses["/predict/compare"][sym]?.quantum_simulator;
          const quantum_real = responses["/predict/compare"][sym]?.quantum_real;
          // Find the max length of y_test arrays
          const maxLen = Math.max(
            classical?.y_test?.length || 0,
            quantum_sim?.y_test?.length || 0,
            quantum_real?.y_test?.length || 0
          );
          // Build combined data array
          const chartData = Array.from({ length: maxLen }).map((_, i) => ({
            date: classical?.dates?.[i] || quantum_sim?.dates?.[i] || quantum_real?.dates?.[i] || i,
            Classical: classical?.y_pred?.[i],
            QuantumSimulator: quantum_sim?.y_pred?.[i],
            QuantumReal: quantum_real?.y_pred?.[i],
            Actual: classical?.y_test?.[i] || quantum_sim?.y_test?.[i] || quantum_real?.y_test?.[i],
          }));
          return (
            <div key={sym}>
              {/* Tabular View */}
              <h3 className="text-lg font-semibold mt-4 mb-2">Tabular Comparison for {sym}</h3>
              <table className="w-full text-xs bg-slate-800 rounded mb-4">
                <thead>
                  <tr>
                    <th className="p-2">Model</th>
                    <th className="p-2">MSE</th>
                    <th className="p-2">y_pred (first 5)</th>
                  </tr>
                </thead>
                <tbody>
                  {['classical', 'quantum_simulator', 'quantum_real'].map(type => (
                    <tr key={type}>
                      <td className="p-2 font-bold">{type.replace('_', ' ').toUpperCase()}</td>
                      <td className="p-2">{responses["/predict/compare"][sym]?.[type]?.mse ?? '-'}</td>
                      <td className="p-2">{JSON.stringify(responses["/predict/compare"][sym]?.[type]?.y_pred?.slice(0,5) ?? [])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {/* Graphical View */}
              <h3 className="text-lg font-semibold mb-2">Graphical Comparison for {sym}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <XAxis dataKey="date" hide />
                  <YAxis />
                  <Tooltip />
                  <CartesianGrid strokeDasharray="3 3" />
                  {classical?.y_pred && <Line type="monotone" dataKey="Classical" stroke="#4ade80" dot={false} />}
                  {quantum_sim?.y_pred && <Line type="monotone" dataKey="QuantumSimulator" stroke="#818cf8" dot={false} />}
                  {quantum_real?.y_pred && <Line type="monotone" dataKey="QuantumReal" stroke="#f472b6" dot={false} />}
                  {classical?.y_test && <Line type="monotone" dataKey="Actual" stroke="#fbbf24" dot={false} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </motion.div>
    </div>
  );
};

export default Dashboard;
