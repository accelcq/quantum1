#!/bin/bash

# Backend Health Check Script
echo "🔍 Quantum1 Backend Health Check"
echo "================================="

# Get backend URL from your deployment
BACKEND_URL="http://f3ea7191-us-south.lb.appdomain.cloud:8080"

echo "Backend URL: $BACKEND_URL"
echo ""

# Test 1: Basic connectivity
echo "1️⃣ Testing basic connectivity..."
curl -v --connect-timeout 10 --max-time 30 "$BACKEND_URL" 2>&1 | head -20

echo ""
echo "2️⃣ Testing health endpoint..."
curl -v --connect-timeout 10 --max-time 30 "$BACKEND_URL/health" 2>&1 | head -20

echo ""
echo "3️⃣ Testing version endpoint..."
curl -v --connect-timeout 10 --max-time 30 "$BACKEND_URL/version" 2>&1 | head -20

echo ""
echo "4️⃣ Testing docs endpoint..."
curl -I --connect-timeout 10 --max-time 30 "$BACKEND_URL/docs" 2>&1 | head -10

echo ""
echo "5️⃣ Testing with different user agent..."
curl -H "User-Agent: HealthCheck/1.0" --connect-timeout 10 --max-time 30 "$BACKEND_URL/health" 2>&1 | head -10

echo ""
echo "🔍 Health check complete"