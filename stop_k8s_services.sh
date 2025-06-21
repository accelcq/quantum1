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

# Remove backend and frontend images for this tag
for img in quantum1-backend quantum1-frontend; do
  full_image="$REGISTRY/$NAMESPACE/$img:$IMAGE_TAG"
  echo "🧹 Attempting to remove image: $full_image"
  ibmcloud cr image-rm "$full_image" || echo "(Image $full_image not found or already deleted)"
done

echo "✅ Image cleanup complete."
