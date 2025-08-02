#!/bin/bash

echo "🔧 Fixing Docker Build Issue and Redeploying"
echo "============================================"

echo "Step 1: Verify files are in place..."
echo "Checking if requirements.txt exists in app directory:"
if [ -f "app/requirements.txt" ]; then
    echo "✅ app/requirements.txt exists"
else
    echo "❌ app/requirements.txt missing"
    exit 1
fi

echo "Checking if Dockerfile is fixed:"
if grep -q "COPY requirements.txt /app/requirements.txt" app/Dockerfile; then
    echo "✅ Dockerfile COPY command is correct"
else
    echo "❌ Dockerfile COPY command needs fixing"
    exit 1
fi

echo ""
echo "Step 2: Commit and push all fixes..."
git add .
git commit -m "Fix Docker build: uncomment requirements.txt COPY and add requirements.txt to app directory"
git push origin main

echo ""
echo "Step 3: Monitor GitHub Actions..."
echo "Visit: https://github.com/$(git remote get-url origin | sed 's|https://github.com/||' | sed 's|.git||')/actions"

echo ""
echo "Step 4: Watch for successful build..."
echo "The previous error was:"
echo "ERROR: Could not open requirements file: [Errno 2] No such file or directory: '/app/requirements.txt'"
echo ""
echo "This should now be fixed because:"
echo "1. ✅ Uncommented COPY requirements.txt line in Dockerfile"
echo "2. ✅ Added requirements.txt to app/ directory"
echo "3. ✅ Fixed qiskit import in main.py"

echo ""
echo "Step 5: Monitor pod restart after successful build..."
echo "kubectl get pods -n quantum1space -l app=quantum1-backend -w"

echo ""
echo "✅ All fixes applied!"
echo "Expected timeline:"
echo "  - GitHub Actions build: 3-5 minutes"
echo "  - Pod restart: 1-2 minutes after build"
echo "  - Backend healthy: 6-8 minutes total"