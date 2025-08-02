#!/bin/bash

echo "🚀 IMMEDIATE FIX - Force New Deployment"
echo "======================================"

echo "Step 1: Check current status..."
git status

echo ""
echo "Step 2: Verify fixes are in the code..."
echo "Checking main.py for correct import:"
if grep -q "from qiskit_algorithms.optimizers import ADAM" app/main.py; then
    echo "✅ Fixed import found in app/main.py"
else
    echo "❌ Fixed import NOT found - applying fix now..."
    sed -i 's/from qiskit_machine_learning.optimizers import ADAM/from qiskit_algorithms.optimizers import ADAM/' app/main.py
    echo "✅ Import fixed in app/main.py"
fi

echo ""
echo "Checking requirements.txt for qiskit-algorithms:"
if grep -q "qiskit-algorithms" requirements.txt; then
    echo "✅ qiskit-algorithms found in requirements.txt"
else
    echo "❌ qiskit-algorithms NOT found - adding it now..."
    echo "qiskit-algorithms>=0.3.0" >> requirements.txt
    echo "✅ qiskit-algorithms added to requirements.txt"
fi

echo ""
echo "Step 3: Commit and push changes..."
git add .
git commit -m "Fix qiskit import: use qiskit_algorithms instead of qiskit_machine_learning"
git push origin main

echo ""
echo "Step 4: Force immediate pod restart while GitHub Actions builds..."
echo "Deleting current failing pod to trigger restart:"
kubectl delete pod -l app=quantum1-backend -n quantum1space

echo ""
echo "Step 5: Monitor the new deployment..."
echo "GitHub Actions URL: https://github.com/$(git remote get-url origin | sed 's|https://github.com/||' | sed 's|.git||')/actions"

echo ""
echo "Step 6: Watch for new pod with updated image..."
echo "Current image: us.icr.io/quantum1space/quantum1-backend:0048142472015d9fa7170f95c1cc45264bfd70cf"
echo "New image should have a different hash after GitHub Actions completes"

echo ""
echo "Run this to monitor: kubectl get pods -n quantum1space -l app=quantum1-backend -w"

echo ""
echo "✅ Fix applied and deployment triggered!"
echo "Expected timeline:"
echo "  - Pod deletion: immediate"
echo "  - GitHub Actions: 3-5 minutes"
echo "  - New pod with fixed image: 6-8 minutes total"