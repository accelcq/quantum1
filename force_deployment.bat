@echo off
REM Windows Batch Script - Force New Deployment
echo 🚀 FORCE NEW DEPLOYMENT - Windows Version
echo ==========================================

set NAMESPACE=quantum1space
set DEPLOYMENT=quantum1-backend

echo ❌ Current Issue: Pod stuck in ImagePullBackOff with old image
echo ✅ Solution: Force new deployment with latest fixed code
echo.

echo Step 1: Delete the failing pod...
kubectl delete pod -l app=quantum1-backend -n %NAMESPACE%

echo.
echo Step 2: Check current deployment image...
kubectl get deployment %DEPLOYMENT% -n %NAMESPACE% -o jsonpath="{.spec.template.spec.containers[0].image}"
echo.

echo.
echo Step 3: Commit and push latest changes to trigger new build...
git add .
git commit -m "Force new deployment: fixed requirements.txt with qiskit-algorithms==0.3.1"
git push origin main

echo.
echo Step 4: Monitor GitHub Actions...
echo GitHub Actions URL: https://github.com/YOUR_USERNAME/quantum1/actions
echo Wait for: Build and push backend Docker image (3-5 minutes)

echo.
echo Step 5: Commands to monitor progress:
echo =====================================
echo Watch pod status:
echo kubectl get pods -n %NAMESPACE% -l app=quantum1-backend -w
echo.
echo Check deployment image:
echo kubectl get deployment %DEPLOYMENT% -n %NAMESPACE% -o jsonpath="{.spec.template.spec.containers[0].image}"
echo.
echo Test health when ready:
echo curl http://f3ea7191-us-south.lb.appdomain.cloud:8080/health

echo.
echo ✅ DEPLOYMENT TRIGGERED
echo Expected timeline:
echo - Pod deletion: immediate
echo - GitHub Actions: 3-5 minutes  
echo - New pod with fixed image: 8-10 minutes
echo - Backend healthy: 10-12 minutes

pause