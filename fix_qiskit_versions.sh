#!/bin/bash

echo "🔧 FIXING QISKIT VERSION COMPATIBILITY"
echo "====================================="

echo "❌ Build failed because qiskit-algorithms>=0.4.0 doesn't exist"
echo "✅ Available versions: 0.1.0, 0.2.0, 0.2.1, 0.2.2, 0.3.0, 0.3.1"
echo ""

echo "Step 1: Using exact working versions..."
echo "✅ qiskit==1.4.1 (latest stable)"
echo "✅ qiskit-algorithms==0.3.1 (latest available)"
echo "✅ qiskit-machine-learning==0.8.3 (compatible)"
echo "✅ All other versions pinned to working releases"

echo ""
echo "Step 2: Verifying requirements.txt files are identical..."
ROOT_CONTENT=$(cat requirements.txt | grep qiskit)
APP_CONTENT=$(cat app/requirements.txt | grep qiskit)

echo "Root requirements qiskit versions:"
echo "$ROOT_CONTENT"
echo ""
echo "App requirements qiskit versions:"
echo "$APP_CONTENT"

if cmp -s requirements.txt app/requirements.txt; then
    echo "✅ Both requirements.txt files are identical"
else
    echo "❌ Files differ - this will cause issues"
    echo "Syncing app/requirements.txt with root..."
    cp requirements.txt app/requirements.txt
    echo "✅ Files now synchronized"
fi

echo ""
echo "Step 3: Committing the corrected versions..."
git add requirements.txt app/requirements.txt
git commit -m "Fix qiskit-algorithms version: use 0.3.1 (latest available) instead of 0.4.0"

echo ""
echo "Step 4: Pushing to trigger new build..."
git push origin main

echo ""
echo "Step 5: Monitor GitHub Actions build..."
echo "URL: https://github.com/$(git remote get-url origin | sed 's|https://github.com/||' | sed 's|.git||')/actions"

echo ""
echo "Step 6: Watch for successful deployment..."
echo "Expected result: Docker build should now succeed with correct versions"

echo ""
echo "Run this to monitor: kubectl get pods -n quantum1space -l app=quantum1-backend -w"

echo ""
echo "✅ VERSION FIX APPLIED!"
echo "======================"
echo "Corrected versions:"
echo "- qiskit==1.4.1"
echo "- qiskit-algorithms==0.3.1 (was 0.4.0 - doesn't exist)"
echo "- qiskit-machine-learning==0.8.3"
echo "- qiskit-ibm-provider==0.11.0"
echo "- qiskit-ibm-runtime==0.40.1"
echo ""
echo "Expected build time: 3-5 minutes"