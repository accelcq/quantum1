#!/bin/bash
# deploy_k8s.sh from accelcq.com
# This script deploys the Quantum1 application to IBM Cloud Kubernetes Service.

set -e

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Load .env-style secrets if available
if [ -f .env.local ]; then
  set -a
  . ./.env.local
  set +a
  log ".env.local loaded"
fi

# Check required environment variables
REQUIRED_VARS=(IBM_CLOUD_API_KEY IBM_CLOUD_REGION IBM_CLOUD_RESOURCE_GROUP IBM_CLOUD_NAMESPACE)
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    log "ERROR: Required environment variable $var is not set."
    exit 1
  fi
done

K8S_CLUSTER_NAME="quantum1-cluster"
log "Using existing Kubernetes cluster: $K8S_CLUSTER_NAME"

log "IBM Cloud login..."
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"

log "Targeting resource group: $IBM_CLOUD_RESOURCE_GROUP"
ibmcloud target -g "$IBM_CLOUD_RESOURCE_GROUP"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  log "ERROR: 'jq' is required for JSON parsing but not found. Please install jq."
  exit 1
fi

# Configure kubectl access for the existing cluster
if ! ibmcloud ks cluster config --cluster "$K8S_CLUSTER_NAME"; then
  log "ERROR: Failed to configure kubectl for cluster $K8S_CLUSTER_NAME"
  exit 1
fi

# Check cluster status
if ! kubectl get nodes >/dev/null 2>&1; then
  log "❌ Failed to access Kubernetes cluster $K8S_CLUSTER_NAME. Please check your IBM Cloud context and permissions."
  exit 1
fi
log "Kubernetes cluster $K8S_CLUSTER_NAME is accessible."

# Ensure container registry namespace
log "Checking for IBM Cloud Container Registry namespace..."
if ! ibmcloud cr namespace-list | grep -q "$IBM_CLOUD_NAMESPACE"; then
  log "Namespace $IBM_CLOUD_NAMESPACE not found. Creating..."
  if ibmcloud cr namespace-add "$IBM_CLOUD_NAMESPACE"; then
    log "Namespace $IBM_CLOUD_NAMESPACE created."
  else
    log "ERROR: Failed to create namespace $IBM_CLOUD_NAMESPACE."
    exit 1
  fi
else
  log "Namespace $IBM_CLOUD_NAMESPACE exists."
fi

log "Logging in to IBM Cloud Container Registry..."
ibmcloud cr login

# Build and push Docker image
IMAGE_NAME="us.icr.io/$IBM_CLOUD_NAMESPACE/quantum1:latest"
log "Building Docker image: $IMAGE_NAME"
docker build -t quantum1 .

log "Tagging Docker image: $IMAGE_NAME"
docker tag quantum1 "$IMAGE_NAME"

log "Pushing Docker image to IBM Cloud Container Registry..."
docker push "$IMAGE_NAME"
log "Docker image $IMAGE_NAME pushed successfully."

# Kubernetes deployment
log "Deploying Kubernetes resources using quantum1.yaml..."
K8S_NAMESPACE="${K8S_NAMESPACE:-accelcqnamespace}"
log "Target Kubernetes namespace: $K8S_NAMESPACE"

if ! kubectl get namespace "$K8S_NAMESPACE" >/dev/null 2>&1; then
  log "Creating Kubernetes namespace: $K8S_NAMESPACE"
  kubectl create namespace "$K8S_NAMESPACE"
else
  log "Namespace $K8S_NAMESPACE already exists."
fi

DEPLOYMENT_FILE="quantum1_k8s.yaml"
log "Creating Kubernetes deployment and service from $DEPLOYMENT_FILE..."
if [ ! -f "$DEPLOYMENT_FILE" ]; then
  log "ERROR: $DEPLOYMENT_FILE not found. Please ensure it exists in the current directory."
  exit 1
fi

kubectl apply -f "$DEPLOYMENT_FILE" -n "$K8S_NAMESPACE"

log "Waiting for deployment rollout..."
kubectl rollout status deployment/quantum1-deployment -n "$K8S_NAMESPACE"

log "Fetching service info..."
kubectl get svc quantum1-service -n "$K8S_NAMESPACE"

EXTERNAL_IP=""
log "Waiting for external IP assignment..."
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl get svc quantum1-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' -n "$K8S_NAMESPACE")
  if [ -z "$EXTERNAL_IP" ]; then
    log "Still waiting for external IP..."
    sleep 10
  fi
done

log "Quantum1 service available at: http://$EXTERNAL_IP:8080"
echo "Quantum1 service URL: http://$EXTERNAL_IP:8080"
echo "Access the service at: http://$EXTERNAL_IP:8080"
log "Deployment completed successfully!"