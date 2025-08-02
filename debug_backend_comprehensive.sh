#!/bin/bash

# Comprehensive IBM Cloud Backend Debugging Script
# This script will help diagnose why the backend is not responding

echo "🔍 IBM Cloud Quantum1 Backend Debugging Guide"
echo "=============================================="
echo "Backend URL: http://f3ea7191-us-south.lb.appdomain.cloud:8080"
echo "Frontend URL: http://3dc94c6e-us-south.lb.appdomain.cloud"
echo ""

# Configuration
NAMESPACE="quantum1space"
BACKEND_SERVICE="quantum1-backend-service"
BACKEND_DEPLOYMENT="quantum1-backend"
BACKEND_URL="http://f3ea7191-us-south.lb.appdomain.cloud:8080"

echo "🔧 STEP 1: Verify IBM Cloud CLI and kubectl configuration"
echo "--------------------------------------------------------"

# Check IBM Cloud CLI login
echo "1.1 Checking IBM Cloud CLI login status..."
ibmcloud target || {
    echo "❌ IBM Cloud CLI not logged in. Please run:"
    echo "   ibmcloud login --apikey YOUR_API_KEY -r us-south"
    exit 1
}

# Check kubectl configuration
echo ""
echo "1.2 Checking kubectl configuration..."
kubectl config current-context || {
    echo "❌ kubectl not configured. Please run:"
    echo "   ibmcloud ks cluster config --cluster quantum1-cluster"
    exit 1
}

# Verify namespace access
echo ""
echo "1.3 Verifying namespace access..."
kubectl get ns $NAMESPACE || {
    echo "❌ Cannot access namespace $NAMESPACE"
    exit 1
}

echo "✅ IBM Cloud and kubectl configuration verified"
echo ""

echo "🏗️ STEP 2: Check Kubernetes resources status"
echo "---------------------------------------------"

echo "2.1 Checking all pods in namespace..."
kubectl get pods -n $NAMESPACE -o wide

echo ""
echo "2.2 Checking deployment status..."
kubectl get deployment $BACKEND_DEPLOYMENT -n $NAMESPACE

echo ""
echo "2.3 Checking service status..."
kubectl get svc $BACKEND_SERVICE -n $NAMESPACE

echo ""
echo "2.4 Checking service endpoints..."
kubectl get endpoints $BACKEND_SERVICE -n $NAMESPACE

echo ""

echo "📋 STEP 3: Detailed resource inspection"
echo "---------------------------------------"

echo "3.1 Describing backend deployment..."
kubectl describe deployment $BACKEND_DEPLOYMENT -n $NAMESPACE

echo ""
echo "3.2 Describing backend service..."
kubectl describe svc $BACKEND_SERVICE -n $NAMESPACE

echo ""
echo "3.3 Getting backend pod details..."
BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l app=quantum1-backend -o jsonpath='{.items[0].metadata.name}')
if [ -n "$BACKEND_POD" ]; then
    echo "Backend pod found: $BACKEND_POD"
    kubectl describe pod $BACKEND_POD -n $NAMESPACE
else
    echo "❌ No backend pod found!"
fi

echo ""

echo "📜 STEP 4: Check backend logs"
echo "-----------------------------"

if [ -n "$BACKEND_POD" ]; then
    echo "4.1 Current backend logs (last 100 lines)..."
    kubectl logs $BACKEND_POD -n $NAMESPACE --tail=100
    
    echo ""
    echo "4.2 Previous backend logs (if pod restarted)..."
    kubectl logs $BACKEND_POD -n $NAMESPACE --previous --tail=50 || echo "No previous logs available"
else
    echo "❌ Cannot get logs - no backend pod found"
fi

echo ""

echo "🔐 STEP 5: Check secrets and environment variables"
echo "--------------------------------------------------"

echo "5.1 Checking if secrets exist..."
kubectl get secret quantum1-secrets -n $NAMESPACE || echo "❌ Secrets not found"

echo ""
echo "5.2 Verifying secret keys..."
kubectl get secret quantum1-secrets -n $NAMESPACE -o jsonpath='{.data}' | jq -r 'keys[]' 2>/dev/null || echo "Cannot read secret keys"

if [ -n "$BACKEND_POD" ]; then
    echo ""
    echo "5.3 Checking environment variables in pod..."
    kubectl exec -n $NAMESPACE $BACKEND_POD -- printenv | grep -E "(FMP_API_KEY|IBM_CLOUD_API_KEY|IBMQ_API_TOKEN)" | sed 's/=.*/=***REDACTED***/' || echo "❌ Cannot check environment variables"
fi

echo ""

echo "🌐 STEP 6: Network connectivity tests"
echo "------------------------------------"

echo "6.1 Testing external connectivity to backend..."
curl -v --connect-timeout 10 --max-time 20 "$BACKEND_URL/health" 2>&1 | head -20

echo ""
echo "6.2 Testing root endpoint..."
curl -v --connect-timeout 10 --max-time 20 "$BACKEND_URL/" 2>&1 | head -20

if [ -n "$BACKEND_POD" ]; then
    echo ""
    echo "6.3 Testing internal pod connectivity..."
    kubectl exec -n $NAMESPACE $BACKEND_POD -- curl -s localhost:8080/health 2>/dev/null || echo "❌ Internal connectivity test failed"
    
    echo ""
    echo "6.4 Checking if backend process is running in pod..."
    kubectl exec -n $NAMESPACE $BACKEND_POD -- ps aux | grep -E "(python|uvicorn|main)" || echo "❌ Backend process not found"
    
    echo ""
    echo "6.5 Checking pod resource usage..."
    kubectl top pod $BACKEND_POD -n $NAMESPACE || echo "Metrics not available"
fi

echo ""

echo "🔄 STEP 7: Check recent deployments and events"
echo "----------------------------------------------"

echo "7.1 Recent events in namespace..."
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -20

echo ""
echo "7.2 Deployment rollout history..."
kubectl rollout history deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE

echo ""
echo "7.3 Current deployment status..."
kubectl rollout status deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE --timeout=10s || echo "Deployment not ready"

echo ""

echo "📊 STEP 8: Compare with frontend (working service)"
echo "-------------------------------------------------"

echo "8.1 Frontend service status..."
kubectl get svc quantum1-frontend-service -n $NAMESPACE || echo "Frontend service not found"

echo ""
echo "8.2 Frontend pod status..."
kubectl get pods -n $NAMESPACE -l app=quantum1-frontend

echo ""

echo "🩺 STEP 9: Diagnostic summary and recommendations"
echo "------------------------------------------------"

# Check if backend pod exists and is running
BACKEND_POD_STATUS=$(kubectl get pods -n $NAMESPACE -l app=quantum1-backend -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
BACKEND_POD_READY=$(kubectl get pods -n $NAMESPACE -l app=quantum1-backend -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)

echo "Backend pod status: ${BACKEND_POD_STATUS:-NOT_FOUND}"
echo "Backend pod ready: ${BACKEND_POD_READY:-NOT_FOUND}"

echo ""
echo "💡 COMMON ISSUES AND FIXES:"
echo "----------------------------"

if [ "$BACKEND_POD_STATUS" != "Running" ]; then
    echo "❌ ISSUE: Backend pod is not running"
    echo "   FIX: kubectl rollout restart deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE"
elif [ "$BACKEND_POD_READY" != "True" ]; then
    echo "❌ ISSUE: Backend pod is running but not ready"
    echo "   FIX: Check logs for startup errors"
else
    echo "❌ ISSUE: Backend pod appears healthy but not responding"
    echo "   FIX: Check service configuration and port binding"
fi

echo ""
echo "🔧 QUICK FIXES TO TRY:"
echo "1. Restart backend deployment:"
echo "   kubectl rollout restart deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE"
echo ""
echo "2. Check and recreate secrets:"
echo "   kubectl delete secret quantum1-secrets -n $NAMESPACE"
echo "   # Then recreate with correct values"
echo ""
echo "3. Scale down and up:"
echo "   kubectl scale deployment/$BACKEND_DEPLOYMENT --replicas=0 -n $NAMESPACE"
echo "   kubectl scale deployment/$BACKEND_DEPLOYMENT --replicas=1 -n $NAMESPACE"
echo ""
echo "4. Re-deploy from GitHub Actions:"
echo "   git commit --allow-empty -m 'trigger deployment'"
echo "   git push origin main"

echo ""
echo "🔍 Debugging complete. Check the output above for issues."