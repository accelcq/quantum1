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

# Function to delete old images, keeping the last N and always deleting the failed IMAGE_TAG if present
cleanup_images() {
  local repo=$1
  echo "🧹 Checking images for $repo..."
  # List images sorted by creation date (oldest last)
  images=$(ibmcloud cr images --format json | jq -r \
    --arg repo "$REGISTRY/$NAMESPACE/$repo" \
    '.[] | select(.repository==$repo) | {tag: .tag, created: .created, digest: .digest} | @base64' | sort)
  # Get tags sorted by creation (oldest last)
  tags=( $(echo "$images" | base64 --decode | jq -r '.tag') )
  digests=( $(echo "$images" | base64 --decode | jq -r '.digest') )
  total=${#tags[@]}
  # Always delete the failed IMAGE_TAG if present
  for i in "${!tags[@]}"; do
    if [[ "${tags[$i]}" == "$IMAGE_TAG" ]]; then
      echo "🧹 Deleting failed image: $repo:${tags[$i]}"
      ibmcloud cr image-rm "$REGISTRY/$NAMESPACE/$repo:${tags[$i]}" || echo "(Image $repo:${tags[$i]} not found or already deleted)"
    fi
  done
  # Delete images older than the last $KEEP_N (excluding IMAGE_TAG)
  if (( total > KEEP_N )); then
    for i in $(seq 0 $((total-KEEP_N-1))); do
      if [[ "${tags[$i]}" != "$IMAGE_TAG" ]]; then
        echo "🧹 Deleting old image: $repo:${tags[$i]}"
        ibmcloud cr image-rm "$REGISTRY/$NAMESPACE/$repo:${tags[$i]}" || echo "(Image $repo:${tags[$i]} not found or already deleted)"
      fi
    done
  fi
}

cleanup_images quantum1-backend
cleanup_images quantum1-frontend

echo "✅ Image cleanup complete."
