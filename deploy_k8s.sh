#!/bin/bash
# deploy_k8s.sh from accelcq.com
# Deploys Quantum1 app to IBM Cloud Kubernetes cluster using Docker Hub image.

set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Load environment variables
if [ -f .env.local ]; then
  set -a
  . ./.env.local
  set +a
  log "🔑 Loaded .env.local"
fi

# Required environment variables
REQUIRED_VARS=(IBM_CLOUD_API_KEY IBM_CLOUD_REGION IBM_CLOUD_RESOURCE_GROUP)
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var:-}" ]; then
    log "❌ ERROR: Required environment variable $var is not set."
    exit 1
  fi
done

K8S_CLUSTER_NAME="quantum1-cluster"
K8S_NAMESPACE="${K8S_NAMESPACE:-accelcqnamespace}"
DEPLOYMENT_FILE="quantum1_k8s.yaml"
IMAGE_NAME="docker.io/ranjantx/quantum1:latest"

# IBM Cloud CLI login
log "🔐 Logging into IBM Cloud..."
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud target -g "$IBM_CLOUD_RESOURCE_GROUP"

log "📡 Configuring cluster access for: $K8S_CLUSTER_NAME"
ibmcloud ks cluster config --cluster "$K8S_CLUSTER_NAME"

kubectl get nodes >/dev/null || {
  log "❌ ERROR: Cannot access Kubernetes cluster $K8S_CLUSTER_NAME"
  exit 1
}
log "✅ Connected to cluster: $K8S_CLUSTER_NAME"

# Namespace setup
if ! kubectl get namespace "$K8S_NAMESPACE" >/dev/null 2>&1; then
  log "📁 Creating namespace: $K8S_NAMESPACE"
  kubectl create namespace "$K8S_NAMESPACE"
else
  log "📁 Namespace exists: $K8S_NAMESPACE"
fi

# Clean up existing deployment & pods
log "🧹 Cleaning up existing deployment and pods (if any)..."
kubectl delete deployment quantum1-deployment -n "$K8S_NAMESPACE" --ignore-not-found
kubectl delete pod -l app=quantum1 -n "$K8S_NAMESPACE" --ignore-not-found

# Docker image handling
log "🐳 Building Docker image..."
docker build -t quantum1 .

log "🐳 Tagging image as: $IMAGE_NAME"
docker tag quantum1 "$IMAGE_NAME"

log "🚀 Pushing image to Docker Hub..."
docker push "$IMAGE_NAME"

# Apply Kubernetes manifests
if [ ! -f "$DEPLOYMENT_FILE" ]; then
  log "❌ ERROR: $DEPLOYMENT_FILE not found."
  exit 1
fi

log "📦 Applying Kubernetes deployment from $DEPLOYMENT_FILE"
kubectl apply -f "$DEPLOYMENT_FILE" -n "$K8S_NAMESPACE"

# Wait and diagnose if rollout takes too long
ROLLOUT_TIMEOUT=120  # seconds
SLEEP_INTERVAL=10
elapsed=0
log "🔄 Waiting for rollout..."

while true; do
  STATUS=$(kubectl get deployment quantum1-deployment -n "$K8S_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
  if [ "$STATUS" == "True" ]; then
    log "✅ Deployment completed successfully."
    break
  fi

  if [ "$elapsed" -ge "$ROLLOUT_TIMEOUT" ]; then
    log "❌ Deployment rollout timeout after ${ROLLOUT_TIMEOUT}s. Gathering diagnostics..."

    log "🔍 Pod status:"
    kubectl get pods -n "$K8S_NAMESPACE"

    for pod in $(kubectl get pods -n "$K8S_NAMESPACE" -o name); do
      log "🔎 Describing $pod"
      kubectl describe "$pod" -n "$K8S_NAMESPACE" || true

      log "📄 Logs from $pod"
      kubectl logs "$pod" -n "$K8S_NAMESPACE" || true

      log "📄 Previous logs from $pod (if available)"
      kubectl logs "$pod" -n "$K8S_NAMESPACE" --previous || true
    done

    exit 1
  fi

  sleep "$SLEEP_INTERVAL"
  elapsed=$((elapsed + SLEEP_INTERVAL))
  log "⌛ Still waiting... (${elapsed}s elapsed)"
done
# Ensure service is created
log "🔧 Ensuring service is created..."

# Post-deployment diagnostics
log "📋 Checking pod status..."
kubectl get pods -l app=quantum1 -n "$K8S_NAMESPACE"

FAILED_POD=$(kubectl get pods -l app=quantum1 -n "$K8S_NAMESPACE" --field-selector=status.phase!=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$FAILED_POD" ]; then
  log "⚠️ Pod $FAILED_POD is not running. Showing describe/logs:"
  kubectl describe pod "$FAILED_POD" -n "$K8S_NAMESPACE" || true
  kubectl logs "$FAILED_POD" -n "$K8S_NAMESPACE" || true
fi

# Expose external IP
log "🌐 Retrieving service IP..."
kubectl get svc quantum1-service -n "$K8S_NAMESPACE"

EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl get svc quantum1-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' -n "$K8S_NAMESPACE")
  [ -z "$EXTERNAL_IP" ] && log "⏳ Waiting for external IP..." && sleep 10
done

log "✅ Deployment successful!"
echo "🌍 Quantum1 available at: http://$EXTERNAL_IP:8080"
log "📖 For more details, visit: https://accelcq.com/docs/quantum1-deployment or contact ranjan@accelcq.com"
log "🚀 Deployment completed successfully!"
# Cleanup local Docker image
log "🧼 Cleaning up local Docker image..."
docker rmi "$IMAGE_NAME" || true
log "🧼 Local Docker image cleanup complete."
log "🎉 All tasks completed successfully!"
# Exit script
exit 0
# End of deploy_k8s.sh
# This script deploys the Quantum1 application to an IBM Cloud Kubernetes cluster.
# It builds the Docker image, pushes it to Docker Hub, and applies the Kubernetes manifests.