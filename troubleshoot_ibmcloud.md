# IBM Cloud Backend Troubleshooting Guide

## Step 1: Login to IBM Cloud
```bash
# Login using your API key
ibmcloud login --apikey YOUR_IBM_CLOUD_API_KEY

# Or login interactively
ibmcloud login
```

## Step 2: Set target region and resource group
```bash
# List available regions
ibmcloud regions

# Set target region (example: us-south)
ibmcloud target -r us-south

# List resource groups
ibmcloud resource groups

# Set resource group
ibmcloud target -g default
```

## Step 3: Check Container Registry access
```bash
# Login to Container Registry
ibmcloud cr login

# List namespaces
ibmcloud cr namespaces

# Check images (if any)
ibmcloud cr images
```

## Step 4: Check Kubernetes cluster status
```bash
# List clusters
ibmcloud ks clusters

# Get cluster config (replace CLUSTER_NAME with your actual cluster name)
ibmcloud ks cluster config --cluster CLUSTER_NAME

# Check kubectl connection
kubectl cluster-info

# Check pods status
kubectl get pods

# Check services
kubectl get services

# Check deployments
kubectl get deployments
```

## Step 5: Check application logs
```bash
# Get pod logs (replace POD_NAME with actual pod name)
kubectl logs POD_NAME

# Follow logs in real-time
kubectl logs -f POD_NAME

# Check all pods in namespace
kubectl get pods --all-namespaces
```

## Step 6: Common Issues and Fixes

### Issue 1: API Key Authentication
If login fails with API key:
```bash
# Verify API key is correct
echo $IBM_CLOUD_API_KEY

# Try refreshing the login
ibmcloud login --apikey $IBM_CLOUD_API_KEY --no-region
```

### Issue 2: Container Registry Issues
```bash
# Re-login to container registry
ibmcloud cr login

# Check registry quotas
ibmcloud cr quota

# Check registry info
ibmcloud cr info
```

### Issue 3: Kubernetes Connection Issues
```bash
# Refresh cluster config
ibmcloud ks cluster config --cluster YOUR_CLUSTER_NAME --admin

# Check cluster health
ibmcloud ks cluster get --cluster YOUR_CLUSTER_NAME

# Reset kubectl context
kubectl config current-context
kubectl config get-contexts
```

### Issue 4: Application Not Responding
```bash
# Check if application is running
kubectl get pods -l app=quantum1-backend

# Restart deployment
kubectl rollout restart deployment/quantum1-backend

# Check service endpoints
kubectl get endpoints

# Port forward for local testing
kubectl port-forward service/quantum1-backend 8080:8080
```

## Step 7: Backend Server Specific Checks

### Check Environment Variables
```bash
# Check if pod has correct environment variables
kubectl describe pod POD_NAME

# Check secrets
kubectl get secrets
kubectl describe secret quantum1-secrets
```

### Check Application Health
```bash
# Test health endpoint directly
curl http://YOUR_BACKEND_URL/health

# Test connectivity endpoint
curl http://YOUR_BACKEND_URL/test-connectivity

# Check if API keys are set correctly
kubectl exec POD_NAME -- env | grep API_KEY
```

## Step 8: Fix Common Backend Issues

### Update deployment with correct API keys
```bash
# Edit deployment
kubectl edit deployment quantum1-backend

# Or apply updated config
kubectl apply -f deployment.yaml
```

### Restart services
```bash
# Delete and recreate pods
kubectl delete pod POD_NAME

# Scale deployment down and up
kubectl scale deployment quantum1-backend --replicas=0
kubectl scale deployment quantum1-backend --replicas=1
```

## Step 9: Local Development Fix

If you're running locally, check:

### Environment Variables (.env.local)
```bash
# Check if .env.local file exists and has correct values
cat .env.local

# Example content should be:
# FMP_API_KEY=your_financial_modeling_prep_key
# IBM_CLOUD_API_KEY=your_ibm_cloud_key
# IBM_QUANTUM_API_TOKEN=your_quantum_token
```

### Python Dependencies
```bash
# Install/update dependencies
pip install -r requirements.txt

# Check if all imports work
python -c "from app.main import app; print('Imports successful')"
```

### Start Local Server
```bash
# Start FastAPI server locally
cd c:\Users\RanjanKumar\Projects\Qiskit\qiskit_100_py311
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Step 10: Check Frontend Configuration

### Update API URL in frontend
Check if the frontend is pointing to the correct backend URL:

```javascript
// In App2.js or Dashboard.js
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8080";
```

### Test CORS settings
The backend should have correct CORS settings for your frontend domain.

## Quick Diagnostic Commands Summary:
```bash
# 1. Login and setup
ibmcloud login --apikey YOUR_API_KEY
ibmcloud target -r us-south -g default

# 2. Check cluster
ibmcloud ks clusters
ibmcloud ks cluster config --cluster YOUR_CLUSTER

# 3. Check application
kubectl get all
kubectl logs -l app=quantum1-backend

# 4. Test connectivity
curl http://YOUR_BACKEND_URL/health

# 5. Local development
cd c:\Users\RanjanKumar\Projects\Qiskit\qiskit_100_py311
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Run these commands step by step and let me know what errors you encounter!