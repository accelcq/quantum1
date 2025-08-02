# IBM Cloud Backend Debugging - Manual Steps
# ==========================================

## STEP 1: Setup IBM Cloud CLI and kubectl
```bash
# Login to IBM Cloud
ibmcloud login --apikey YOUR_API_KEY -r us-south

# Configure kubectl
ibmcloud ks cluster config --cluster quantum1-cluster

# Verify namespace access
kubectl get ns quantum1space
```

## STEP 2: Check Pod Status
```bash
# Check all pods
kubectl get pods -n quantum1space

# Check backend pod specifically
kubectl get pods -n quantum1space -l app=quantum1-backend

# Get pod details
kubectl describe pods -n quantum1space -l app=quantum1-backend
```

## STEP 3: Check Backend Logs
```bash
# Get current logs
kubectl logs -l app=quantum1-backend -n quantum1space --tail=100

# Get previous logs (if pod restarted)
kubectl logs -l app=quantum1-backend -n quantum1space --previous --tail=50

# Follow logs in real-time
kubectl logs -l app=quantum1-backend -n quantum1space -f
```

## STEP 4: Check Service Configuration
```bash
# Check service status
kubectl get svc quantum1-backend-service -n quantum1space

# Describe service
kubectl describe svc quantum1-backend-service -n quantum1space

# Check endpoints
kubectl get endpoints quantum1-backend-service -n quantum1space
```

## STEP 5: Check Deployment Status
```bash
# Check deployment
kubectl get deployment quantum1-backend -n quantum1space

# Describe deployment
kubectl describe deployment quantum1-backend -n quantum1space

# Check rollout status
kubectl rollout status deployment/quantum1-backend -n quantum1space
```

## STEP 6: Test Internal Connectivity
```bash
# Get pod name
BACKEND_POD=$(kubectl get pods -n quantum1space -l app=quantum1-backend -o jsonpath='{.items[0].metadata.name}')

# Test internal health check
kubectl exec -n quantum1space $BACKEND_POD -- curl -s localhost:8080/health

# Check if process is running
kubectl exec -n quantum1space $BACKEND_POD -- ps aux | grep python

# Check environment variables
kubectl exec -n quantum1space $BACKEND_POD -- printenv | grep API_KEY
```

## STEP 7: Check Secrets
```bash
# Check if secrets exist
kubectl get secret quantum1-secrets -n quantum1space

# Check secret keys
kubectl get secret quantum1-secrets -n quantum1space -o jsonpath='{.data}' | jq -r 'keys[]'
```

## STEP 8: Common Quick Fixes
```bash
# Restart deployment
kubectl rollout restart deployment/quantum1-backend -n quantum1space

# Scale down and up
kubectl scale deployment/quantum1-backend --replicas=0 -n quantum1space
sleep 30
kubectl scale deployment/quantum1-backend --replicas=1 -n quantum1space

# Delete and recreate pod
kubectl delete pod -l app=quantum1-backend -n quantum1space
```

## STEP 9: External Connectivity Test
```bash
# Test health endpoint
curl -v http://f3ea7191-us-south.lb.appdomain.cloud:8080/health

# Test root endpoint
curl -v http://f3ea7191-us-south.lb.appdomain.cloud:8080/

# Test with timeout
curl --connect-timeout 10 --max-time 30 http://f3ea7191-us-south.lb.appdomain.cloud:8080/health
```

## STEP 10: Check Recent Events
```bash
# Check recent events
kubectl get events -n quantum1space --sort-by='.lastTimestamp' | tail -20

# Check deployment events
kubectl describe deployment quantum1-backend -n quantum1space | grep -A 10 Events:
```