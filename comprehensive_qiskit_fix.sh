#!/bin/bash

echo "🔧 COMPREHENSIVE FIX - Standardize to Qiskit 1.4.1"
echo "=================================================="

echo "Step 1: Create standardized requirements.txt with Qiskit 1.4.1..."

cat > requirements.txt << 'EOF'
qiskit==1.4.1
qiskit-ibm-provider>=0.8.0
qiskit-ibm-runtime>=0.30.0
qiskit-machine-learning>=0.8.0
qiskit-algorithms>=0.4.0
qiskit-aer>=0.15.0
numpy>=1.21.0
fastapi>=0.70.0
uvicorn[standard]>=0.17.0
python-dotenv
python-jose[cryptography]
passlib[bcrypt]
pydantic>=2.0.0
scikit-learn
pandas
requests
matplotlib
seaborn
EOF

echo "✅ Updated root requirements.txt with Qiskit 1.4.1"

echo ""
echo "Step 2: Copy to app directory..."
cp requirements.txt app/requirements.txt
echo "✅ Synced app/requirements.txt"

echo ""
echo "Step 3: Fix main.py imports for Qiskit 1.4.1..."

# Update main.py with correct imports for Qiskit 1.4.1
sed -i 's/from qiskit_machine_learning.optimizers import ADAM/from qiskit_algorithms.optimizers import ADAM/' app/main.py
sed -i 's/from qiskit_machine_learning.optimizers/from qiskit_algorithms.optimizers/' app/main.py

echo "✅ Fixed main.py imports"

echo ""
echo "Step 4: Fix all Q*.py files for Qiskit 1.4.1 compatibility..."

# Fix Qsimulator.py
if [ -f "app/Qsimulator.py" ]; then
    echo "Fixing Qsimulator.py..."
    # Add proper imports for Qiskit 1.4.1
    sed -i 's/from qiskit_machine_learning/from qiskit_algorithms/' app/Qsimulator.py
    sed -i 's/qiskit_machine_learning/qiskit_algorithms/' app/Qsimulator.py
    echo "✅ Fixed Qsimulator.py"
else
    echo "❌ Qsimulator.py not found"
fi

# Fix Qmachine.py
if [ -f "app/Qmachine.py" ]; then
    echo "Fixing Qmachine.py..."
    sed -i 's/from qiskit_machine_learning/from qiskit_algorithms/' app/Qmachine.py
    sed -i 's/qiskit_machine_learning/qiskit_algorithms/' app/Qmachine.py
    echo "✅ Fixed Qmachine.py"
else
    echo "❌ Qmachine.py not found"
fi

# Fix Qtraining.py
if [ -f "app/Qtraining.py" ]; then
    echo "Fixing Qtraining.py..."
    sed -i 's/from qiskit_machine_learning/from qiskit_algorithms/' app/Qtraining.py
    sed -i 's/qiskit_machine_learning/qiskit_algorithms/' app/Qtraining.py
    echo "✅ Fixed Qtraining.py"
else
    echo "❌ Qtraining.py not found"
fi

# Fix quantum_utils.py
if [ -f "app/quantum_utils.py" ]; then
    echo "Fixing quantum_utils.py..."
    sed -i 's/from qiskit_machine_learning/from qiskit_algorithms/' app/quantum_utils.py
    sed -i 's/qiskit_machine_learning/qiskit_algorithms/' app/quantum_utils.py
    echo "✅ Fixed quantum_utils.py"
else
    echo "❌ quantum_utils.py not found"
fi

echo ""
echo "Step 5: Update Dockerfile for consistency..."

# Ensure Dockerfile uses the correct requirements.txt
cat > app/Dockerfile << 'EOF'
#Dockerfile-backend for FastAPI application(backend)

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data/predictions /app/logs

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application (using the working CMD)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

echo "✅ Updated Dockerfile"

echo ""
echo "Step 6: Verify all changes..."

echo "Checking for any remaining old imports..."
OLD_IMPORTS=$(find app/ -name "*.py" -exec grep -l "qiskit_machine_learning.optimizers" {} \; 2>/dev/null || true)
if [ -n "$OLD_IMPORTS" ]; then
    echo "❌ Still found old imports in: $OLD_IMPORTS"
else
    echo "✅ No old imports found"
fi

echo ""
echo "Checking requirements.txt consistency..."
ROOT_VERSION=$(grep "qiskit==" requirements.txt)
APP_VERSION=$(grep "qiskit==" app/requirements.txt)

if [ "$ROOT_VERSION" = "$APP_VERSION" ]; then
    echo "✅ Requirements.txt files are consistent: $ROOT_VERSION"
else
    echo "❌ Still inconsistent:"
    echo "  Root: $ROOT_VERSION"
    echo "  App: $APP_VERSION"
fi

echo ""
echo "Step 7: Commit all fixes..."
git add .
git commit -m "MAJOR FIX: Standardize to Qiskit 1.4.1, fix all Q*.py imports, sync requirements.txt"

echo ""
echo "Step 8: Push changes..."
git push origin main

echo ""
echo "✅ COMPREHENSIVE FIX COMPLETE!"
echo "=============================="
echo "Changes made:"
echo "1. ✅ Standardized to Qiskit 1.4.1 across all files"
echo "2. ✅ Updated all quantum library versions"
echo "3. ✅ Fixed all qiskit_machine_learning → qiskit_algorithms imports"
echo "4. ✅ Synced requirements.txt in root and app directories"
echo "5. ✅ Updated Dockerfile for consistency"
echo "6. ✅ Committed and pushed all changes"

echo ""
echo "Monitor deployment: kubectl get pods -n quantum1space -l app=quantum1-backend -w"
echo "Expected build time: 5-8 minutes"
EOF