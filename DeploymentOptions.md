# Deployment Options: IBM Cloud

This document provides a comprehensive guide to deploying the Quantum1 Dockerized application (FastAPI + Qiskit + React) across multiple IBM Cloud environments:

- IBM Cloud Code Engine (existing setup)
- IBM Cloud Virtual Server for VPC (Compute VMs)
- IBM Cloud Kubernetes Service (IKS)

---

## 🚀 Objective

Run the same Dockerized Qiskit + FastAPI + React app on:
- IBM Cloud Code Engine (current setup)
- IBM Cloud Virtual Server for VPC
- IBM Cloud Kubernetes Cluster

---

## 🧰 Common Preparation

1. **Build and Push Docker Image**

```bash
docker build -t us.icr.io/accelcq/quantum1:latest .
docker push us.icr.io/accelcq/quantum1:latest
```

Make sure you’re authenticated:

```bash
ibmcloud login --apikey $IBM_CLOUD_API_KEY -r us-south
ibmcloud cr login
```

---

## 🏗️ Option 1: IBM Cloud Virtual Server for VPC

### ✅ Step-by-Step

1. **Create VPC Instance**
   - Use Ubuntu
   - Attach Public IP
   - Allow port 8000 in firewall

2. **SSH into VM**
```bash
ssh -i ~/.ssh/your-key.pem root@<VM-IP>
```

3. **Install Docker**
```bash
apt update && apt install -y docker.io
systemctl start docker && systemctl enable docker
```

4. **Run the Docker App**
```bash
docker login us.icr.io
docker pull us.icr.io/accelcq/quantum1:latest
docker run -d -p 8000:8000 us.icr.io/accelcq/quantum1:latest
```

5. **Access the App**
```
http://<VM-IP>:8000/docs
```

---

## ☸️ Option 2: IBM Cloud Kubernetes Service

### ✅ Step-by-Step

1. **Create Cluster**
   - Use IBM Cloud UI to create cluster

2. **Install CLI Tools**
```bash
ibmcloud plugin install container-service
ibmcloud ks cluster config --cluster <CLUSTER_NAME>
```

3. **Create Kubernetes YAML**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum1-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quantum1
  template:
    metadata:
      labels:
        app: quantum1
    spec:
      containers:
      - name: quantum1
        image: us.icr.io/accelcq/quantum1:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: quantum1-service
spec:
  type: LoadBalancer
  selector:
    app: quantum1
  ports:
    - port: 80
      targetPort: 8000
```

4. **Deploy App**
```bash
kubectl apply -f quantum1.yaml
kubectl get svc quantum1-service
```

5. **Access App**
- Use LoadBalancer external IP

---

## 🔍 Comparison Table

| Feature                         | Code Engine                    | Virtual Server (VPC)           | Kubernetes (IKS)                 |
|-------------------------------|--------------------------------|--------------------------------|----------------------------------|
| **Abstraction Level**          | Fully managed (serverless)     | Infrastructure-level (manual)  | Container orchestration          |
| **Scale**                      | Auto-scaled                    | Manual                         | Auto-scaled                      |
| **Startup Speed**              | Cold start (few sec)           | Always-on                      | Fast (pods)                      |
| **Custom OS / Dependencies**   | Limited                        | Full control                   | Full container control           |
| **Best for**                   | APIs, jobs, stateless services | Long-running processes         | Production-scale microservices   |
| **HTTPS/TLS**                  | Built-in                       | Manual (via Nginx or Certbot)  | Built-in via Ingress/LoadBalancer |
| **Cost**                       | Usage-based                    | Pay per hour                   | Per node/hour                    |
| **Complexity**                 | Low                            | Medium                         | High                             |

---

## 🧠 Recommendation

| Use Case                                | Recommended Option         |
|----------------------------------------|----------------------------|
| Rapid API development, small scale     | ✅ Code Engine              |
| Quantum research with custom setups    | ✅ VPC (Compute VM)         |
| Production-grade, scalable deployments | ✅ Kubernetes (IKS)         |

---

For deployment automation, request assistance with Terraform, GitHub Actions, or `quantum1.yaml` templates.
Contact ranjan@accelcq.com for more information.