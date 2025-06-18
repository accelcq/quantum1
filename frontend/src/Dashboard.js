import React, { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const Dashboard = () => {
  const [token, setToken] = useState("");
  const [symbol, setSymbol] = useState("AAPL");
  const [predictionClassical, setPredictionClassical] = useState(null);
  const [predictionQuantum, setPredictionQuantum] = useState(null);
  const [chartData, setChartData] = useState([]);

  const handleLogin = async () => {
    const res = await fetch("/token", { method: "POST" });
    const data = await res.json();
    setToken(data.access_token);
  };

  const predictStock = async () => {
    const headers = { Authorization: `Bearer ${token}` };
    const resClassical = await fetch(`/predict/classical?symbols=${symbol}`, { headers });
    const dataClassical = await resClassical.json();
    setPredictionClassical(dataClassical);

    const resQuantum = await fetch(`/predict/quantum?symbols=${symbol}`, { headers });
    const dataQuantum = await resQuantum.json();
    setPredictionQuantum(dataQuantum);
  };

  const fetchChartData = async () => {
    const res = await fetch(`/historical-data/${symbol}`);
    const data = await res.json();
    setChartData(data.map(day => ({ date: day.date, close: day.close })));
  };

  useEffect(() => {
    fetchChartData();
  }, [symbol]);

  return (
    <div className="p-4 md:p-10 bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen text-white">
      <div className="flex items-center justify-between mb-6 flex-col md:flex-row gap-4">
        <img src="/accelcq-logo.svg" alt="AccelCQ Logo" width={160} height={60} />
        <h1 className="text-3xl font-bold text-center">Quantum1 API Dashboard</h1>
      </div>

      <motion.div className="grid md:grid-cols-2 gap-6 mb-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
        <div className="bg-slate-700 shadow-xl rounded-2xl p-6">
          <h2 className="text-xl font-semibold mb-4">Login</h2>
          <button onClick={handleLogin} className="w-full bg-green-600 py-2 rounded-lg">Login</button>
          <p className="mt-2 text-sm break-all">Token: {token ? `${token.slice(0, 20)}...` : "Not logged in"}</p>
        </div>

        <div className="bg-slate-700 shadow-xl rounded-2xl p-6">
          <h2 className="text-xl font-semibold mb-4">Stock Prediction</h2>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Enter stock symbol" className="w-full px-4 py-2 mb-2 rounded-lg text-black" />
          <button onClick={predictStock} className="w-full bg-blue-600 py-2 rounded-lg">Predict</button>
        </div>
      </motion.div>

      <motion.div className="grid md:grid-cols-2 gap-6 mb-6" initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 1 }}>
        {predictionClassical && (
          <div className="bg-slate-800 shadow-xl rounded-2xl p-6">
            <h2 className="text-xl font-semibold mb-2">Classical Prediction: {symbol}</h2>
            <p className="mb-2">Prediction: <strong>{predictionClassical.prediction}</strong></p>
            <p>Confidence - Up: {predictionClassical.counts?.[1]}%, Down: {predictionClassical.counts?.[0]}%</p>
          </div>
        )}

        {predictionQuantum && (
          <div className="bg-slate-800 shadow-xl rounded-2xl p-6">
            <h2 className="text-xl font-semibold mb-2">Quantum Prediction: {symbol}</h2>
            <p className="mb-2">Prediction: <strong>{predictionQuantum.prediction}</strong></p>
            <p>Confidence - Up: {predictionQuantum.counts?.[1]}%, Down: {predictionQuantum.counts?.[0]}%</p>
          </div>
        )}
      </motion.div>

      <motion.div className="bg-slate-700 rounded-2xl shadow-2xl p-6" initial={{ scale: 0.95 }} animate={{ scale: 1 }} transition={{ duration: 1 }}>
        <h2 className="text-xl font-semibold mb-4">Historical Stock Data Chart: {symbol}</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="date" hide />
            <YAxis domain={['dataMin', 'dataMax']} />
            <Tooltip />
            <CartesianGrid strokeDasharray="3 3" />
            <Line type="monotone" dataKey="close" stroke="#4ade80" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
};

export default Dashboard;
