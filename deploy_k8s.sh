#!/bin/bash
#deploy_k8s.sh from accelcq.com
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
REQUIRED_VARS=(IBM_CLOUD_API_KEY IBM_CLOUD_REGION IBM_CLOUD_RESOURCE_GROUP IBM_CLOUD_NAMESPACE K8S_CLUSTER_NAME)
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    log "ERROR: Required environment variable $var is not set."
    exit 1
  fi
done

log "IBM Cloud login..."
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"

log "Targeting resource group: $IBM_CLOUD_RESOURCE_GROUP"
ibmcloud target -g "$IBM_CLOUD_RESOURCE_GROUP"

log "Checking for container registry namespace: $IBM_CLOUD_NAMESPACE"
if ! ibmcloud cr namespace-list | grep -q "$IBM_CLOUD_NAMESPACE"; then
  log "Namespace $IBM_CLOUD_NAMESPACE not found. Creating..."
  if ibmcloud cr namespace-add "$IBM_CLOUD_NAMESPACE"; then
    log "Namespace $IBM_CLOUD_NAMESPACE created."
  else
    log "ERROR: Failed to create namespace $IBM_CLOUD_NAMESPACE. Check your permissions."
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
log "Tagging Docker image..."
docker tag quantum1 $IMAGE_NAME
log "Pushing Docker image to IBM Cloud Container Registry..."
docker push $IMAGE_NAME

# Kubernetes cluster config
log "Checking for Kubernetes cluster: $K8S_CLUSTER_NAME"
if ! ibmcloud ks cluster ls | grep -q "$K8S_CLUSTER_NAME"; then
  log "Kubernetes cluster $K8S_CLUSTER_NAME not found."
  # Check if user has permission to create clusters
  if ibmcloud ks cluster create --help > /dev/null 2>&1; then
    log "You have permission to create clusters, but this script does not auto-create clusters. Please create it manually if needed."
  else
    log "WARNING: You do NOT have permission to create Kubernetes clusters. Please contact your IBM Cloud admin."
  fi
  exit 1
else
  log "Kubernetes cluster $K8S_CLUSTER_NAME found."
fi

log "Configuring kubectl for cluster: $K8S_CLUSTER_NAME"
ibmcloud ks cluster config --cluster "$K8S_CLUSTER_NAME"

# Apply K8S deployment
log "Applying Kubernetes deployment (quantum1.yaml)..."
kubectl apply -f quantum1.yaml
log "Waiting for deployment rollout..."
kubectl rollout status deployment/quantum1-deployment
log "Getting service info..."
kubectl get svc quantum1-service

# Get the external IP of the service
EXTERNAL_IP=""
log "Waiting for external IP of the service..."
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl get svc quantum1-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  if [ -z "$EXTERNAL_IP" ]; then
    log "Waiting for external IP..."
    sleep 10
  fi
done
log "Quantum1 service is available at http://$EXTERNAL_IP:8080"
echo "Quantum1 service URL: http://$EXTERNAL_IP:8080"
echo "You can access the Quantum1 service at: http://$EXTERNAL_IP:8080"