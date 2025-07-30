# Quantum1 Troubleshooting Guide

This guide helps you resolve common issues with the Quantum1 application.

## Common Errors and Solutions

### 1. IBM Quantum Authentication Error (401 Unauthorized)

**Error Message:**
```
qiskit_ibm_runtime.exceptions.IBMNotAuthorizedError: '401 Client Error: Unauthorized for url: https://auth.quantum.ibm.com/api/users/loginWithToken. Login failed., Error code: 3446.'
```

**Cause:** Invalid or missing IBM Quantum API token.

**Solution:**
1. **Get a valid IBM Quantum API token:**
   - Go to https://quantum.ibm.com/
   - Sign in with your IBM account
   - Navigate to 'Account' → 'API token'
   - Create a new token or copy an existing one

2. **Set up the token using our helper script:**
   ```bash
   cd app
   python setup_quantum_token.py
   ```

3. **Or manually set the environment variable:**
   ```bash
   export IBMQ_API_TOKEN="your_token_here"
   ```

4. **For Kubernetes deployment, update your secrets:**
   ```bash
   kubectl create secret generic quantum-secrets \
     --from-literal=IBMQ_API_TOKEN="your_token_here" \
     --from-literal=FMP_API_KEY="your_fmp_key" \
     --from-literal=IBM_CLOUD_API_KEY="your_ibm_cloud_key"
   ```

5. **Test the connection:**
   ```bash
   python setup_quantum_token.py test
   ```

### 2. Insufficient Data Error

**Error Message:**
```
ValueError: Test data is empty
```

**Cause:** The stock data doesn't have enough samples for meaningful prediction.

**Solution:**
- The application now handles this automatically with fallback predictions
- For better results, try symbols with more historical data (e.g., AAPL, MSFT, GOOGL)
- The system requires at least 10 samples for basic prediction, 450+ for optimal results

### 3. Missing Dependencies

**Error Message:**
```
ImportError: Qiskit Aer is not installed
```

**Solution:**
```bash
pip install qiskit-aer qiskit-machine-learning qiskit-ibm-runtime
```

### 4. Frontend Connection Issues

**Error Message:**
```
Failed to fetch from API
```

**Solution:**
1. Check if the backend is running:
   ```bash
   kubectl get pods -n your-namespace
   ```

2. Check backend logs:
   ```bash
   kubectl logs -n your-namespace your-backend-pod
   ```

3. Verify the API URL in the frontend:
   - Check `frontend/src/App2.js` for the `API_URL` setting
   - Ensure it matches your backend service URL

## Environment Variables

Make sure these environment variables are set:

```bash
# Required for stock data
FMP_API_KEY=your_financial_modeling_prep_api_key

# Required for IBM Quantum access
IBMQ_API_TOKEN=your_ibm_quantum_api_token

# Required for IBM Cloud services
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key
```

## Data Requirements

### Minimum Data Requirements:
- **Basic prediction:** 10+ samples
- **Optimal prediction:** 450+ samples
- **Recommended symbols:** AAPL, MSFT, GOOGL, AMZN, TSLA

### Data Sources:
- Historical data is fetched from Financial Modeling Prep API
- Data is cached locally to reduce API calls
- Cache files are stored in `data/historical/`

## Fallback Behavior

When quantum prediction fails, the system automatically falls back to:
1. **Classical prediction** using mean values
2. **Quantum-inspired simulation** with noise
3. **Error response** with detailed error information

## Logging

Check logs for detailed error information:

```bash
# Backend logs
kubectl logs -n your-namespace your-backend-pod

# Application logs
tail -f app/logs/execution_log.log
```

## Performance Tips

1. **Use cached data:** The system caches stock data daily
2. **Choose popular symbols:** More data available for major stocks
3. **Monitor API limits:** FMP API has rate limits
4. **Use simulator first:** Test with quantum simulator before using real hardware

## Getting Help

If you continue to experience issues:

1. Check the logs for detailed error messages
2. Verify all environment variables are set correctly
3. Test the IBM Quantum connection using the setup script
4. Ensure you have sufficient stock data for your chosen symbols

## Quick Fix Commands

```bash
# Test IBM Quantum connection
python app/setup_quantum_token.py test

# Set up IBM Quantum token
python app/setup_quantum_token.py

# Check backend status
kubectl get pods -n your-namespace

# View backend logs
kubectl logs -n your-namespace your-backend-pod --tail=100

# Restart backend pod
kubectl delete pod -n your-namespace your-backend-pod
``` 