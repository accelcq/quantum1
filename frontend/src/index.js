import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import App from "./App2"; // API Reference Console
import Dashboard from "./Dashboard"; // Quantum1 Dashboard
import "./index.css";

const Intro = () => (
  <div className="p-10 bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen text-white">
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-6 text-center">Welcome to AccelCQ Quantum Demo 1</h1>

      <div className="grid md:grid-cols-2 gap-6 mb-12">
        <div className="bg-slate-800 p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-2">🚀 Quantum1 Dashboard</h2>
          <ul className="list-disc list-inside text-slate-200">
            <li>✅ Secure JWT login and session handling</li>
            <li>📈 Visualize historical data for top US stocks</li>
            <li>🧠 Train Classical ML models and view results</li>
            <li>⚛️ Train Quantum ML models and analyze results</li>
            <li>🔍 Predict using ANN and QNN with symbol filtering</li>
            <li>✅ Headless UI Tabs for organized API layout</li>
            <li>✅ Fully interactive and responsive interface</li>
          </ul>
          <Link to="/Dashboard" className="inline-block mt-4 px-4 py-2 bg-green-600 rounded hover:bg-green-500">Launch Dashboard</Link>
        </div>

        <div className="bg-slate-800 p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-2">📘 API Reference Console</h2>
          <ul className="list-disc list-inside text-slate-200">
            <li>✔️ Manual testing of all API endpoints</li>
            <li>✔️ Login with JWT and make secure requests</li>
            <li>✔️ POST and GET supported for full stack testing</li>
            <li>✔️ JSON/text response viewer</li>
            <li>✔️ Integrated live stock chart</li>
          </ul>
          <Link to="/API-Reference" className="inline-block mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-500">Open API Reference</Link>
        </div>
      </div>

      <p className="text-center text-slate-400 text-sm">Typical Project Structure:</p>
      <pre className="text-xs text-slate-400 bg-slate-900 rounded p-4 overflow-x-auto mt-2">
<div>my-react-app/</div>
<div>├── node_modules/</div>
<div>├── public/</div>
<div>├── src/</div>
<div>│   └── App.js</div>
<div>│   └── Dashboard.js</div>
<div>│   └── index.js</div>
<div>│   └── index.css</div>
<div>├── tailwind.config.js   ✅ to support js,jsx,ts,tsx files</div>
<div>├── postcss.config.js    ✅ to support tailwindcss and autoprefixer</div>
<div>├── package.json</div>
      </pre>
    </div>
  </div>
);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Intro />} />
        <Route path="/Dashboard" element={<Dashboard />} />
        <Route path="/API-Reference" element={<App />} />
        <Route path="*" element={<Intro />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
