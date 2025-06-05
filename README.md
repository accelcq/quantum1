# Qiskit 1.0 Project
# Quantum1: FastAPI + Qiskit API on IBM Cloud

This project provides a full-stack template to build, containerize, and deploy a Qiskit-based API using FastAPI, Docker, and IBM Cloud Code Engine.

---

## 🚀 Project Objective

Build a Python FastAPI application that executes quantum circuits via Qiskit SDK, dockerize the application, and deploy it seamlessly on IBM Cloud Code Engine using GitHub Actions and CLI automation.

---

## 🧰 Prerequisites

### 1. Operating System

* Windows 11 (Home or Pro)

### 2. Accounts & Access

* ✅ GitHub account + Repository: [accelcq/quantum1](https://github.com/accelcq/quantum1)
* ✅ IBM Cloud account with billing enabled: [https://cloud.ibm.com](https://cloud.ibm.com)

  * Create IBM Cloud API Key
  * Enable: `IAM → Manage → Access → API Keys`
  * Create resource group (e.g., `quantum-group`)
  * Create a container registry namespace (e.g., `accelcq`) in `us-south`
  * Create Code Engine project (e.g., `quantum1-project`)
* ✅ IBM Quantum Token: [https://quantum.ibm.com](https://quantum.ibm.com)

### 3. Software Installation

Install the following on your local machine:

| Tool          | Description / Link                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| Docker        | [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)                             |
| Git           | [Install Git](https://git-scm.com/downloads)                                                          |
| Python 3.11   | [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) <br> ✅ Add to PATH! |
| VS Code       | [Download VS Code](https://code.visualstudio.com/)                                                    |
| IBM Cloud CLI | [Install CLI](https://github.com/IBM-Cloud/ibm-cloud-cli-release)                                     |
| Pyenv-win     | [pyenv-win](https://github.com/pyenv-win/pyenv-win) (optional)                                        |

### 4. VS Code Extensions (Manual Installation)

Install via Extensions panel:

* Docker
* Python (by Microsoft)
* Pylance
* GitHub Copilot + Copilot Chat
* GitHub Actions
* IBM Cloud CLI + IBM Cloud Account
* Q by QuLearnLabs
* OpenQASM – Qiskit Debugger

---

## 🛠️ Local Development Environment Setup

1. Clone setup helper:

   ```bash
   git clone https://github.com/ranjantx/qiskit_windows_setup
   ```
2. Use VENV from `C:\Users\<USERNAME>\Envs\Qiskit\qiskit_100_py311`
3. Activate the VENV:

   ```bash
   C:\Users\RanjanKumar\Envs\Qiskit\qiskit_100_py311\Scripts\activate
   ```
4. Open folder in VS Code: `C:\Users\RanjanKumar\Projects\Qiskit\qiskit_100_py311`
5. Set interpreter: `Ctrl+Shift+P → Python: Select Interpreter → qiskit_100_py311`
6. Install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```
7. Terminal should show: `(qiskit_100_py311)` in VS Code

⏱️ **Note:** The entire dev setup process realistically takes 20–30 minutes including account registration, tool installations, and quantum environment setup. The 10-minute estimate assumes tools are preinstalled and keys/tokens already provisioned.

---

## 📁 Project Layout (Quantum1)

```
quantum1/
├── .github/workflows/deploy.yml       # CI/CD workflow
├── .vscode/tasks.json                 # VS Code local task runner
├── app/main.py                        # FastAPI + Qiskit API
├── Dockerfile                         # Container build spec
├── deploy.sh                          # CLI automation script
├── deploy.yaml                        # Code Engine config
├── requirements.txt                   # Qiskit + FastAPI
├── .env.local                         # Secrets (ignored by git)
├── .gitignore                         # Excludes venv, .env, etc.
```

---

## 📦 Python Dependencies (`requirements.txt`)

```text
qiskit==1.0.0
qiskit-ibm-provider>=0.7.0
qiskit-ibm-runtime>=0.23.0
qiskit-machine-learning>=0.7.0
numpy>=1.21.0
fastapi>=0.70.0
uvicorn>=0.17.0
python-dotenv
```

---

## 🐳 Docker Setup

### `Dockerfile`

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build Docker Image",
      "type": "shell",
      "command": "docker build -t us.icr.io/accelcq/quantum1:latest .",
      "problemMatcher": []
    },
    {
      "label": "Push Docker Image",
      "type": "shell",
      "command": "docker push us.icr.io/accelcq/quantum1:latest",
      "problemMatcher": []
    },
    {
      "label": "Run All Locally",
      "dependsOn": ["Build Docker Image", "Push Docker Image"],
      "dependsOrder": "sequence",
      "problemMatcher": []
    }
  ]
}
```

---

## 🔐 Local Secrets (`.env.local`)

```dotenv
IBM_CLOUD_API_KEY="<Paste your API Key>"
IBM_CLOUD_REGION="us-south"
IBM_QUANTUM_API_URL="https://api.quantum-computing.ibm.com/api"
IBM_QUANTUM_API_TOKEN="<Paste your Quantum Token>"
```

---

## 🔑 GitHub Secrets (for deploy.yml)

In GitHub → Settings → Secrets and Variables → Actions:

* `IBM_CLOUD_API_KEY`
* `IBM_CLOUD_REGION`
* `IBM_QUANTUM_API_TOKEN`

---

## 🚀 GitHub Actions Deployment

### `.github/workflows/deploy.yml`

```yaml
name: Deploy to IBM Code Engine
on:
  push:
    branches: [main]
env:
  IMAGE_NAME: us.icr.io/accelcq/quantum1
  IBM_CLOUD_REGION: us-south
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Install IBM Cloud CLI
        run: curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
      - name: IBM Cloud Login
        run: |
          ibmcloud login --apikey "${{ secrets.IBM_CLOUD_API_KEY }}" -r ${{ env.IBM_CLOUD_REGION }}
          ibmcloud cr login
      - name: Build & Push
        run: |
          docker build -t $IMAGE_NAME:latest .
          docker push $IMAGE_NAME:latest
      - name: Deploy to Code Engine
        run: |
          ibmcloud ce project select --name quantum1-project
          ibmcloud ce application apply --file deploy.yaml
```

---

## 🧪 Troubleshooting & Migration Notes

* Use `https://qisk.it/1-0-constraints` for dependency constraints
* Avoid legacy `qiskit-terra`
* Use `qiskit >=1.0.0` in a fresh virtual environment

---
# Quantum1: FastAPI + Qiskit API on IBM Cloud

This project provides a full-stack template to build, containerize, and deploy a Qiskit-based API using FastAPI, Docker, and IBM Cloud Code Engine.

...

## 🧪 How to Run and Test

### ▶️ Run Locally (in VS Code)

1. Activate your virtual environment:

   ```bash
   C:\Users\<UserName>\Envs\Qiskit\qiskit_100_py311\Scripts\activate
   ```

   ⏱ Time: \~3 sec

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   ⏱ Time: \~3 min

3. Run the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

   ⏱ Time: <10 sec (startup)

4. Open in browser:

   * [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * [http://127.0.0.1:8000](http://127.0.0.1:8000)

Expected response:

```json
{"backend": "ibmq_qasm_simulator"}
```

> 🧪 Tip: Ensure your `.env.local` file has a valid `IBM_QUANTUM_API_TOKEN` for remote backend execution.

### ☁️ Run on IBM Cloud (Code Engine)

1. Trigger GitHub Action or run `./deploy.sh`
2. Once deployed, run:

   ```bash
   ibmcloud ce application get --name quantum1 --output url
   ```
3. Open the app URL in your browser

   * Example: `https://quantum1.<hash>.us-south.codeengine.appdomain.cloud`
   * Navigate to `/docs` to test via Swagger

> ✅ Ensure `quantum1-secrets` is created and injected using Code Engine secret environment variable.

---

## ⏱ Time Estimates

| Step                                            | Time Taken |
| ----------------------------------------------- | ---------- |
| GitHub account + repo setup                     | 5–10 min   |
| IBM Cloud setup (billing, IAM, Code Engine)     | 30–60 min  |
| Software install (Docker, Git, Python, VS Code) | 10–15 min  |
| Clone + activate VENV                           | 1 min      |
| VS Code interpreter config                      | 1 min      |
| `pip install -r requirements.txt`               | 3 min      |
| Local run test (`uvicorn`, `/docs`)             | <1 min     |
| Docker build + push                             | 2–3 min    |
| Deploy to Code Engine                           | 2–3 min    |
| GitHub Action full CI/CD                        | \~4–5 min  |

---

For questions or issues, consult Qiskit Slack or raise a GitHub Issue.
