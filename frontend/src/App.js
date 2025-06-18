import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";
import Image from "next/image";

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
        <Image src="/accelcq-logo.svg" alt="AccelCQ Logo" width={160} height={60} />
        <h1 className="text-3xl font-bold text-center">Quantum1 API Dashboard</h1>
      </div>

      <motion.div className="grid md:grid-cols-2 gap-6 mb-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
        <Card className="bg-slate-700 shadow-xl rounded-2xl">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">Login</h2>
            <Button onClick={handleLogin} className="w-full">Login</Button>
            <p className="mt-2 text-sm break-all">Token: {token ? `${token.slice(0, 20)}...` : "Not logged in"}</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-700 shadow-xl rounded-2xl">
          <CardContent className="p-6">
            <h2 className="text-xl font-semibold mb-4">Stock Prediction</h2>
            <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Enter stock symbol" className="mb-2" />
            <Button onClick={predictStock} className="w-full">Predict</Button>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div className="grid md:grid-cols-2 gap-6 mb-6" initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 1 }}>
        {predictionClassical && (
          <Card className="bg-slate-800 shadow-xl rounded-2xl">
            <CardContent className="p-6">
              <h2 className="text-xl font-semibold mb-2">Classical Prediction: {symbol}</h2>
              <p className="mb-2">Prediction: <strong>{predictionClassical.prediction}</strong></p>
              <p>Confidence - Up: {predictionClassical.counts?.[1]}%, Down: {predictionClassical.counts?.[0]}%</p>
            </CardContent>
          </Card>
        )}

        {predictionQuantum && (
          <Card className="bg-slate-800 shadow-xl rounded-2xl">
            <CardContent className="p-6">
              <h2 className="text-xl font-semibold mb-2">Quantum Prediction: {symbol}</h2>
              <p className="mb-2">Prediction: <strong>{predictionQuantum.prediction}</strong></p>
              <p>Confidence - Up: {predictionQuantum.counts?.[1]}%, Down: {predictionQuantum.counts?.[0]}%</p>
            </CardContent>
          </Card>
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
//         <input placeholder="Stock Symbol" value={predictClassicalSymbol} onChange={e => setPredictClassicalSymbol(e.target.value)} />
//         <button onClick={() => post("/predict/classical", { symbols: predictClassicalSymbol }, setPredictionClassical)}>Predict Classical</button>
//         <input placeholder="Stock Symbol" value={predictQuantumSymbol} onChange={e => setPredictQuantumSymbol(e.target.value)} />
//         <button onClick={() => post("/predict/quantum", { symbols: predictQuantumSymbol }, setPredictionQuantum)}>Predict Quantum</button>
//         <h2>Classical Prediction: {predictionClassical?.prediction}</h2>
//         <h2>Quantum Prediction: {predictionQuantum?.prediction}</h2>
//         <h2>Historical Data for {historicalSymbol}</h2>
//         <input placeholder="Stock Symbol" value={historicalSymbol} onChange={e => setHistoricalSymbol(e.target.value)} />
//         <button onClick={() => get(`/historical-data/${historicalSymbol}`, setHistoricalData)}>Fetch Historical Data</button>
//         <pre>{JSON.stringify(historicalData, null, 2)}</pre>
//         <h2>Historical Data Chart for {historicalSymbol}</h2>
//         <ResponsiveContainer width="100%" height={300}>  
//           <LineChart data={historicalData}>
//             <XAxis dataKey="date" />
//             <YAxis />
//             <Tooltip />
//             <CartesianGrid stroke="#f5f5f5" />
//             <Line type="monotone" dataKey="close" stroke="#ff7300" yAxisId={0} />
//           </LineChart>
//         </ResponsiveContainer>
//       </div>
//     </div>
//   );
// };
// export default App;
// import React, { useState } from "react";
// import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
//   ResponsiveContainer } from "recharts";
// import { motion } from "framer-motion";  
// import Image from "next/image";
// import { Card, CardContent } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Input } from "@/components/ui/input";                                                                       
