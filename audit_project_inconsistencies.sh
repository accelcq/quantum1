#!/bin/bash

echo "🔍 COMPREHENSIVE PROJECT AUDIT - Version Inconsistencies"
echo "======================================================="

echo "1️⃣ Checking requirements.txt files across project..."
echo ""
echo "Root requirements.txt:"
if [ -f "requirements.txt" ]; then
    echo "EXISTS - Content:"
    head -10 requirements.txt
else
    echo "❌ MISSING"
fi

echo ""
echo "App requirements.txt:"
if [ -f "app/requirements.txt" ]; then
    echo "EXISTS - Content:"
    head -10 app/requirements.txt
else
    echo "❌ MISSING"
fi

echo ""
echo "2️⃣ Checking Qiskit versions..."
echo "Root requirements.txt qiskit version:"
grep -i "qiskit" requirements.txt || echo "Not found"

echo ""
echo "App requirements.txt qiskit version:"
grep -i "qiskit" app/requirements.txt || echo "Not found"

echo ""
echo "3️⃣ Checking main.py imports..."
echo "Current imports in app/main.py:"
grep -n "from qiskit" app/main.py || echo "No qiskit imports found"
grep -n "import qiskit" app/main.py || echo "No qiskit imports found"

echo ""
echo "4️⃣ Checking Q*.py files for version compatibility issues..."
echo ""
echo "Qsimulator.py imports:"
if [ -f "app/Qsimulator.py" ]; then
    grep -n "from qiskit" app/Qsimulator.py || echo "No qiskit imports found"
    grep -n "import qiskit" app/Qsimulator.py || echo "No qiskit imports found"
else
    echo "❌ Qsimulator.py MISSING"
fi

echo ""
echo "Qmachine.py imports:"
if [ -f "app/Qmachine.py" ]; then
    grep -n "from qiskit" app/Qmachine.py || echo "No qiskit imports found"
    grep -n "import qiskit" app/Qmachine.py || echo "No qiskit imports found"
else
    echo "❌ Qmachine.py MISSING"
fi

echo ""
echo "Qtraining.py imports:"
if [ -f "app/Qtraining.py" ]; then
    grep -n "from qiskit" app/Qtraining.py || echo "No qiskit imports found"
    grep -n "import qiskit" app/Qtraining.py || echo "No qiskit imports found"
else
    echo "❌ Qtraining.py MISSING"
fi

echo ""
echo "quantum_utils.py imports:"
if [ -f "app/quantum_utils.py" ]; then
    grep -n "from qiskit" app/quantum_utils.py || echo "No qiskit imports found"
    grep -n "import qiskit" app/quantum_utils.py || echo "No qiskit imports found"
else
    echo "❌ quantum_utils.py MISSING"
fi

echo ""
echo "5️⃣ Checking for deprecated imports..."
echo "Searching for old qiskit_machine_learning.optimizers imports:"
find . -name "*.py" -exec grep -l "qiskit_machine_learning.optimizers" {} \; || echo "None found (good)"

echo ""
echo "Searching for qiskit_algorithms imports:"
find . -name "*.py" -exec grep -l "qiskit_algorithms" {} \; || echo "None found"

echo ""
echo "6️⃣ Summary of inconsistencies found:"
echo "=================================="

ROOT_QISKIT=$(grep "qiskit==" requirements.txt 2>/dev/null | head -1 || echo "NOT_FOUND")
APP_QISKIT=$(grep "qiskit==" app/requirements.txt 2>/dev/null | head -1 || echo "NOT_FOUND")

echo "Root qiskit version: $ROOT_QISKIT"
echo "App qiskit version: $APP_QISKIT"

if [ "$ROOT_QISKIT" != "$APP_QISKIT" ]; then
    echo "❌ VERSION MISMATCH DETECTED"
else
    echo "✅ Versions match"
fi

echo ""
echo "📋 RECOMMENDED FIXES:"
echo "==================="
echo "1. Standardize on Qiskit 1.4.1 across all files"
echo "2. Update all Q*.py files to use compatible imports"
echo "3. Remove any deprecated qiskit_machine_learning.optimizers imports"
echo "4. Ensure all quantum files use qiskit_algorithms for optimizers"
echo "5. Update requirements.txt in both root and app directories"