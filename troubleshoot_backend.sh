#!/bin/bash

# Backend Troubleshooting Script for IBM Cloud Kubernetes
echo "🔧 Quantum1 Backend Troubleshooting"
echo "===================================="

NAMESPACE="quantum1space"
BACKEND_URL="http://f3ea7191-us-south.lb.appdomain.cloud:8080"

echo "🔍 1. Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=quantum1-backend

echo ""
echo "🔍 2. Checking deployment status..."
kubectl get deployment quantum1-backend -n $NAMESPACE

echo ""
echo "🔍 3. Checking service status..."
kubectl get svc quantum1-backend-service -n $NAMESPACE

echo ""
echo "🔍 4. Describing service (detailed)..."
kubectl describe svc quantum1-backend-service -n $NAMESPACE

echo ""
echo "🔍 5. Checking pod logs (last 50 lines)..."
kubectl logs -l app=quantum1-backend -n $NAMESPACE --tail=50

echo ""
echo "🔍 6. Describing pod (events)..."
kubectl describe pod -l app=quantum1-backend -n $NAMESPACE

echo ""
echo "🔍 7. Testing backend connectivity..."
echo "Trying health endpoint: $BACKEND_URL/health"
curl -v --connect-timeout 15 --max-time 30 "$BACKEND_URL/health" || echo "❌ Health endpoint failed"

echo ""
echo "Trying root endpoint: $BACKEND_URL/"
curl -v --connect-timeout 15 --max-time 30 "$BACKEND_URL/" || echo "❌ Root endpoint failed"

echo ""
echo "🔍 8. Checking if backend pod is ready..."
BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l app=quantum1-backend -o jsonpath='{.items[0].metadata.name}')
if [ -n "$BACKEND_POD" ]; then
  echo "Backend pod: $BACKEND_POD"
  echo "Pod ready status:"
  kubectl get pod $BACKEND_POD -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
  echo ""
  
  echo "Testing internal connectivity from pod:"
  kubectl exec -n $NAMESPACE $BACKEND_POD -- curl -s localhost:8080/health || echo "❌ Internal health check failed"
else
  echo "❌ No backend pod found"
fi

echo ""
echo "🔍 9. Checking secrets..."
kubectl get secret quantum1-secrets -n $NAMESPACE || echo "❌ Secrets not found"

echo ""
echo "🔧 Troubleshooting complete"
echo ""
echo "💡 Common fixes:"
echo "   1. If pod is not ready: Check logs for startup errors"
echo "   2. If connectivity fails: Check service configuration"
echo "   3. If secrets missing: Re-run secret creation step"
echo "   4. If pod crashes: Check resource limits and dependencies"