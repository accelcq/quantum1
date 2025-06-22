#!/bin/bash
# stop_k8s_services.sh
# Safely deletes frontend and backend deployments and services for a clean redeploy.
# Also removes the corresponding images from IBM Cloud Container Registry and checks quota.
# Usage: IMAGE_TAG=<sha> bash stop_k8s_services.sh

set -e

# Load environment variables
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
  echo "🔑 Loaded environment variables from .env.local"
fi

# Set defaults if not set
NAMESPACE="${IBM_CLOUD_NAMESPACE:-quantum1space}"
REGION="${IBM_CLOUD_REGION:-us-south}"
REGISTRY="us.icr.io"
KEEP_N=1  # Number of most recent images to keep per repo

# IMAGE_TAG must be set (commit SHA from CI/CD or manual export)
if [ -z "$IMAGE_TAG" ]; then
  echo "❌ IMAGE_TAG is not set. Please set IMAGE_TAG to the commit SHA used for deployment."
  echo "   Example: IMAGE_TAG=abcdef123 bash stop_k8s_services.sh"
  exit 1
fi

echo "🆔 Using IMAGE_TAG: $IMAGE_TAG"

# Delete backend deployment and service
kubectl delete deployment quantum1-backend-deployment || true
kubectl delete service quantum1-backend-service || true

# Delete frontend deployment and service
kubectl delete -f frontend/quantum1-frontend-deployment.yaml || true
kubectl delete -f frontend/quantum1-frontend-service.yaml || true

echo "✅ All specified deployments and services deleted (if they existed)."

# IBM Cloud login and registry login
if [ -z "$IBM_CLOUD_API_KEY" ]; then
  echo "Error: IBM_CLOUD_API_KEY is not set."
  exit 1
fi
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$REGION"
ibmcloud cr login

echo "🔎 Checking IBM Cloud Container Registry quota..."
ibmcloud cr quota

print_debug_info() {
  echo "\n==== IBM Cloud Info ===="
  ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$REGION" -g "${IBM_CLOUD_RESOURCE_GROUP:-$RESOURCE_GROUP}"
  ibmcloud cr login
  ibmcloud ks cluster config --cluster "${K8S_CLUSTER_NAME:-quantum1-cluster}"

  echo "\n==== Kubernetes Cluster Info ===="
  kubectl get nodes
  kubectl get pods -n "$NAMESPACE" -o wide
  kubectl get svc -n "$NAMESPACE"
  kubectl describe deployment quantum1-frontend -n "$NAMESPACE"
  kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp

  echo "\n==== IBM Cloud Container Registry Image Digests ===="
  for img in quantum1-frontend quantum1-backend; do
    echo "\nImage digests for $img:"
    ibmcloud cr image-digests --restrict "$NAMESPACE/$img"
  done

  echo "\n==== Pod Logs (frontend) ===="
  frontend_pods=$(kubectl get pods -n "$NAMESPACE" -l app=quantum1-frontend -o jsonpath='{.items[*].metadata.name}')
  for pod in $frontend_pods; do
    echo "\nLogs for pod: $pod"
    kubectl logs "$pod" -n "$NAMESPACE" || echo "(No logs for $pod)"
  done
}

# Function to delete old images, keeping the last N and always deleting the failed IMAGE_TAG if present
cleanup_images() {
  local repo=$1
  echo "🧹 Checking images for $repo..."
  images_json=$(ibmcloud cr images --format json)
  # Get all tags for this repo, sorted by created date (newest first)
  tags=( $(echo "$images_json" | jq -r --arg repo "$REGISTRY/$NAMESPACE/$repo" \
    '.[] | select(.repository==$repo) | .tag' | tac) )
  total=${#tags[@]}
  deleted=0
  # Always delete the failed IMAGE_TAG if present
  for tag in "${tags[@]}"; do
    if [[ "$tag" == "$IMAGE_TAG" ]]; then
      echo "🧹 Deleting failed image: $repo:$tag"
      ibmcloud cr image-rm "$REGISTRY/$NAMESPACE/$repo:$tag" || echo "(Image $repo:$tag not found or already deleted)"
      ((deleted++))
    fi
  done
  # Delete images older than the last $KEEP_N (excluding IMAGE_TAG)
  if (( total > KEEP_N )); then
    for ((i=0; i<total-KEEP_N; i++)); do
      tag="${tags[$i]}"
      if [[ "$tag" != "$IMAGE_TAG" ]]; then
        echo "🧹 Deleting old image: $repo:$tag"
        ibmcloud cr image-rm "$REGISTRY/$NAMESPACE/$repo:$tag" || echo "(Image $repo:$tag not found or already deleted)"
        ((deleted++))
      fi
    done
  fi
  echo "🧹 Deleted $deleted images for $repo."
}

print_debug_info
cleanup_images quantum1-backend
print_debug_info
cleanup_images quantum1-frontend

echo "✅ Image cleanup complete."
