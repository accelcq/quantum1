#!/bin/bash

# Force New Deployment Script
echo "🚀 Force New Deployment with Fixed Code"
echo "======================================="

echo "📝 Step 1: Check current git status..."
git status

echo ""
echo "📝 Step 2: Add and commit all changes..."
git add .
git commit -m "Fix qiskit import error - use qiskit_algorithms.optimizers instead of qiskit_machine_learning.optimizers"

echo ""
echo "📝 Step 3: Push to trigger new deployment..."
git push origin main

echo ""
echo "📝 Step 4: Monitor GitHub Actions..."
echo "Visit: https://github.com/$(git config --get remote.origin.url | sed 's|https://github.com/||' | sed 's|.git||')/actions"

echo ""
echo "📝 Step 5: While waiting, you can also force a new image build..."
echo "Current image being used: us.icr.io/quantum1space/quantum1-backend:0048142472015d9fa7170f95c1cc45264bfd70cf"
echo "New deployment should create a new image with latest git commit hash"

echo ""
echo "📝 Step 6: Monitor pod restart..."
echo "kubectl get pods -n quantum1space -l app=quantum1-backend -w"

echo ""
echo "✅ Deployment triggered. Expected timeline:"
echo "   - GitHub Actions build: 3-5 minutes"
echo "   - Pod restart: 1-2 minutes after build"
echo "   - Total time: 5-7 minutes"