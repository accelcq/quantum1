# PowerShell Script - Force New Deployment
Write-Host "🚀 FORCE NEW DEPLOYMENT - PowerShell Version" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

$NAMESPACE = "quantum1space"
$DEPLOYMENT = "quantum1-backend"

Write-Host "❌ Current Issue: Pod stuck in ImagePullBackOff with old image" -ForegroundColor Red
Write-Host "✅ Solution: Force new deployment with latest fixed code" -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Delete the failing pod..." -ForegroundColor Yellow
kubectl delete pod -l app=quantum1-backend -n $NAMESPACE

Write-Host ""
Write-Host "Step 2: Check current deployment image..." -ForegroundColor Yellow
$currentImage = kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}'
Write-Host "Current image: $currentImage" -ForegroundColor Cyan

Write-Host ""
Write-Host "Step 3: Commit and push latest changes..." -ForegroundColor Yellow
git add .
git commit -m "Force new deployment: fixed requirements.txt with qiskit-algorithms==0.3.1"
git push origin main

Write-Host ""
Write-Host "Step 4: Monitor GitHub Actions..." -ForegroundColor Yellow
$repoUrl = git remote get-url origin
$repoPath = $repoUrl -replace "https://github.com/", "" -replace ".git", ""
Write-Host "GitHub Actions URL: https://github.com/$repoPath/actions" -ForegroundColor Cyan
Write-Host "Wait for: Build and push backend Docker image (3-5 minutes)" -ForegroundColor White

Write-Host ""
Write-Host "Step 5: Commands to monitor progress:" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow
Write-Host "Watch pod status:" -ForegroundColor White
Write-Host "kubectl get pods -n $NAMESPACE -l app=quantum1-backend -w" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check deployment image:" -ForegroundColor White
Write-Host "kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test health when ready:" -ForegroundColor White
Write-Host "curl http://f3ea7191-us-south.lb.appdomain.cloud:8080/health" -ForegroundColor Cyan

Write-Host ""
Write-Host "✅ DEPLOYMENT TRIGGERED" -ForegroundColor Green
Write-Host "Expected timeline:" -ForegroundColor White
Write-Host "- Pod deletion: immediate" -ForegroundColor White
Write-Host "- GitHub Actions: 3-5 minutes" -ForegroundColor White
Write-Host "- New pod with fixed image: 8-10 minutes" -ForegroundColor White
Write-Host "- Backend healthy: 10-12 minutes" -ForegroundColor White

Write-Host ""
Write-Host "Press any key to continue monitoring..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")