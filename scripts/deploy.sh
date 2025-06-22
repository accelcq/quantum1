# scripts/deploy.sh
# Unified deployment script for GitHub Actions with health checks and rollback

set -e

# Required environment variables
# IMAGE_TAG        - Tag used for image versions (e.g., commit SHA)
# NAMESPACE        - Kubernetes namespace
# REGION           - IBM Cloud region (e.g., us-south)
# RESOURCE_GROUP   - IBM Cloud resource group
# CLUSTER_NAME     - Kubernetes cluster name
# IBM_CLOUD_API_KEY - IBM Cloud API key
# IBMQ_API_TOKEN    - IBM Q Experience token (optional for app runtime)

if [ -z "$IMAGE_TAG" ]; then
  echo "❌ IMAGE_TAG is not set. Use: IMAGE_TAG=<sha> bash scripts/deploy.sh"
  exit 1
fi

NAMESPACE="quantum1space"
echo "📦 Deploying frontend and backend using IMAGE_TAG=$IMAGE_TAG..."

# Apply deployments and services
cat <<EOF | envsubst | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum1-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quantum1-backend
  template:
    metadata:
      labels:
        app: quantum1-backend
    spec:
      containers:
        - name: quantum1-backend
          image: us.icr.io/quantum1space/quantum1-backend:\${IMAGE_TAG}
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: quantum1-backend-service
spec:
  type: LoadBalancer
  selector:
    app: quantum1-backend
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum1-frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quantum1-frontend
  template:
    metadata:
      labels:
        app: quantum1-frontend
    spec:
      containers:
        - name: quantum1-frontend
          image: us.icr.io/quantum1space/quantum1-frontend:\${IMAGE_TAG}
          ports:
            - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: quantum1-frontend-service
spec:
  type: LoadBalancer
  selector:
    app: quantum1-frontend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
EOF

# Wait for backend service external IP
echo "⏳ Waiting for backend service external IP..."
for i in {1..30}; do
  BACKEND_IP=$(kubectl get svc quantum1-backend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
  if [ ! -z "$BACKEND_IP" ]; then
    echo "✅ Backend IP: $BACKEND_IP"
    break
  fi
  echo "Waiting... ($i)"; sleep 5
done

if [ -z "$BACKEND_IP" ]; then
  echo "❌ Failed to get backend service IP"
  exit 1
fi

# Restart deployments to ensure updates
kubectl rollout restart deployment/quantum1-frontend -n $NAMESPACE
kubectl rollout restart deployment/quantum1-backend -n $NAMESPACE

# Rollout status checks
set +e
kubectl rollout status deployment/quantum1-frontend -n $NAMESPACE || kubectl rollout undo deployment/quantum1-frontend -n $NAMESPACE
kubectl rollout status deployment/quantum1-backend -n $NAMESPACE || kubectl rollout undo deployment/quantum1-backend -n $NAMESPACE
set -e

# Health checks
FRONTEND_IP=$(kubectl get svc quantum1-frontend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "🌐 Frontend: http://$FRONTEND_IP"
echo "🌐 Backend: http://$BACKEND_IP:8080/docs"

echo "🔎 Health checking backend..."
curl -sSf http://$BACKEND_IP:8080/health || echo "❌ Backend health check failed"

echo "🔎 Health checking frontend..."
curl -sSf http://$FRONTEND_IP || echo "❌ Frontend health check failed"

# Debugging
kubectl get svc -n $NAMESPACE

if [ $? -ne 0 ]; then
  echo "⚠️ Printing backend logs..."
  kubectl logs -l app=quantum1-backend -n $NAMESPACE || true
  echo "⚠️ Printing frontend logs..."
  kubectl logs -l app=quantum1-frontend -n $NAMESPACE || true
  echo "⚠️ Describing all pods..."
  kubectl describe pods -n $NAMESPACE || true
fi

echo "✅ Quantum1 Deployment Complete."
echo "🚀 Frontend accessible at: http://$FRONTEND_IP"
echo "🔗 Backend API Docs: http://$BACKEND_IP:8080/docs"
