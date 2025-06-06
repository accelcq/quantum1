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

# Ensure VPC and Subnet setup
log "VPC_NAME: ${VPC_NAME:-not set}"
log "SUBNET_NAME: ${SUBNET_NAME:-not set}"
log "GATEWAY_NAME: ${GATEWAY_NAME:-not set}"
log "ZONE: ${ZONE:-not set}"
if [ -z "$VPC_NAME" ] || [ -z "$SUBNET_NAME" ] || [ -z "$GATEWAY_NAME" ] || [ -z "$ZONE" ]; then
  log "ERROR: VPC_NAME, SUBNET_NAME, GATEWAY_NAME, and ZONE must be set in the environment."
  exit 1
fi
# Ensure IBM Cloud VPC plugin is installed
if ! ibmcloud plugin list | grep -q 'vpc-infrastructure'; then
  log "IBM Cloud VPC plugin not found. Installing..."
  if ! ibmcloud plugin install vpc-infrastructure -f; then
    log "ERROR: Failed to install IBM Cloud VPC plugin."
    exit 1
  fi
else
  log "IBM Cloud VPC plugin is already installed."
fi

# Check if VPC and Subnet exist, create if not
log "Checking for VPC and Subnet..."
if ! ibmcloud is vpc "$VPC_NAME" >/dev/null 2>&1; then
  log "Creating VPC: $VPC_NAME"
  if ! ibmcloud is vpc-create "$VPC_NAME"; then
    log "ERROR: Failed to create VPC $VPC_NAME"
    exit 1
  fi
else
  log "VPC $VPC_NAME already exists."
fi

if ! ibmcloud is subnet "$SUBNET_NAME" >/dev/null 2>&1; then
  log "Creating subnet: $SUBNET_NAME"
  # Correct argument order: --vpc VPC_NAME --zone ZONE --ipv4-address-count 256 --name SUBNET_NAME
  if ! ibmcloud is subnet-create "$VPC_NAME" "$ZONE" --ipv4-address-count 256 --name "$SUBNET_NAME"; then
    log "ERROR: Failed to create subnet $SUBNET_NAME"
    exit 1
  fi
else
  log "Subnet $SUBNET_NAME already exists."
fi

if ! ibmcloud is public-gateway "$GATEWAY_NAME" >/dev/null 2>&1; then
  log "Creating public gateway: $GATEWAY_NAME"
  if ! ibmcloud is public-gateway-create "$ZONE" "$VPC_NAME" --name "$GATEWAY_NAME"; then
    log "ERROR: Failed to create public gateway $GATEWAY_NAME"
    exit 1
  fi
  if ! ibmcloud is subnet-update "$SUBNET_NAME" --public-gateway "$GATEWAY_NAME"; then
    log "ERROR: Failed to attach public gateway $GATEWAY_NAME to subnet $SUBNET_NAME"
    exit 1
  fi
else
  log "Public gateway $GATEWAY_NAME already exists."
fi

# Kubernetes cluster setup
if ! ibmcloud oc cluster get --cluster "$K8S_CLUSTER_NAME" >/dev/null 2>&1; then
  log "Creating Kubernetes cluster: $K8S_CLUSTER_NAME"
  ibmcloud oc cluster create vpc-gen2 \
    --name "$K8S_CLUSTER_NAME" \
    --vpc "$VPC_NAME" \
    --zone "$ZONE" \
    --flavor bx2.4x16 \
    --workers 2 \
    --subnet "$SUBNET_NAME"
else
  log "Kubernetes cluster $K8S_CLUSTER_NAME already exists."
fi

# Configure kubectl access
ibmcloud oc cluster config --cluster "$K8S_CLUSTER_NAME"

# Check cluster status
if ! kubectl cluster-info > /dev/null 2>&1; then
  echo "❌ Failed to access Kubernetes cluster $K8S_CLUSTER_NAME"
  exit 1
fi
log "Kubernetes cluster $K8S_CLUSTER_NAME is accessible."

# Check and create container registry namespace
log "Checking for IBM Cloud Container Registry namespace..."
if [ -z "$IBM_CLOUD_NAMESPACE" ]; then
  log "ERROR: IBM_CLOUD_NAMESPACE is not set. Please set it in your environment."
  exit 1
fi

# Ensure the namespace exists (single check)
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
log "Building Docker image: $IMAGE_NAME,  cmd=docker build -t quantum1 ."
if ! docker build -t quantum1 .; then
  log "ERROR: Docker build failed. Please check your Dockerfile and environment."
  exit 1
fi
log "Tagging Docker image..., cmd = docker tag quantum1 $IMAGE_NAME"
if ! docker tag quantum1 "$IMAGE_NAME"; then
  log "ERROR: Failed to tag Docker image $IMAGE_NAME. Check your Docker setup."
  exit 1
fi
log "Docker image tagged successfully: $IMAGE_NAME"
log "Pushing Docker image to IBM Cloud Container Registry..., cmd=docker push $IMAGE_NAME"
if ! docker push "$IMAGE_NAME"; then
  log "ERROR: Failed to push Docker image $IMAGE_NAME. Check your permissions and network."
  exit 1
fi
log "Docker image $IMAGE_NAME pushed successfully."

# Create Kubernetes deployment and service
log "Creating Kubernetes deployment and service from quantum1.yaml..."
if [ ! -f quantum1.yaml ]; then
  log "ERROR: quantum1.yaml not found. Please ensure it exists in the current directory."
  exit 1
fi

# Kubernetes cluster config
log "Checking for Kubernetes cluster: $K8S_CLUSTER_NAME"
if [ -z "$K8S_CLUSTER_NAME" ]; then
  log "ERROR: K8S_CLUSTER_NAME is not set. Please set it in your environment."
# Kubernetes cluster config
log "Checking for Kubernetes cluster: $K8S_CLUSTER_NAME"
if [ -z "$K8S_CLUSTER_NAME" ]; then
  log "ERROR: K8S_CLUSTER_NAME is not set. Please set it in your environment."
  exit 1
fi
K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
log "Using Kubernetes namespace: $K8S_NAMESPACE"
if ! kubectl get namespace "$K8S_NAMESPACE" >/dev/null 2>&1; then
  log "Creating Kubernetes namespace: $K8S_NAMESPACE"
  kubectl create namespace "$K8S_NAMESPACE"
else
  log "Kubernetes namespace $K8S_NAMESPACE already exists."
fi
log "Waiting for deployment rollout..."
kubectl rollout status deployment/quantum1-deployment -n "$K8S_NAMESPACE"
log "Getting service info..."
kubectl get svc quantum1-service  -n "$K8S_NAMESPACE"

# Get the external IP of the service
EXTERNAL_IP=""
log "Waiting for external IP of the service..."
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl get svc quantum1-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' -n "$K8S_NAMESPACE")
  if [ -z "$EXTERNAL_IP" ]; then
    log "Waiting for external IP..."
    sleep 10
  fi
done
log "Quantum1 service is available at http://$EXTERNAL_IP:8080"
echo "Quantum1 service URL: http://$EXTERNAL_IP:8080"
echo "You can access the Quantum1 service at: http://$EXTERNAL_IP:8080"