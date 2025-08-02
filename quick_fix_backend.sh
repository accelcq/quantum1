#!/bin/bash

# Quick Fix Script for Backend Issues
echo "🚑 Quick Fix Script for Backend Issues"
echo "======================================"

NAMESPACE="quantum1space"
BACKEND_DEPLOYMENT="quantum1-backend"

echo "🔧 Attempting common fixes..."

echo ""
echo "1️⃣ Restarting backend deployment..."
kubectl rollout restart deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE

echo ""
echo "2️⃣ Waiting for deployment to be ready..."
kubectl rollout status deployment/$BACKEND_DEPLOYMENT -n $NAMESPACE --timeout=300s

echo ""
echo "3️⃣ Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=quantum1-backend

echo ""
echo "4️⃣ Testing health endpoint..."
sleep 30  # Give backend time to start
curl --connect-timeout 10 --max-time 30 http://f3ea7191-us-south.lb.appdomain.cloud:8080/health || echo "❌ Still not responding"

echo ""
echo "5️⃣ If still not working, checking logs..."
kubectl logs -l app=quantum1-backend -n $NAMESPACE --tail=20

echo ""
echo "✅ Quick fix attempt complete. If backend is still not responding, run the full debug script."