#!/bin/bash
# run_stop_k8s_services_ci.sh
# Wrapper to run stop_k8s_services.sh with IMAGE_TAG from CI/CD
set -e

# Pass IMAGE_TAG from environment (should be set by workflow)
if [ -z "$IMAGE_TAG" ]; then
  echo "❌ IMAGE_TAG is not set."
  exit 1
fi

bash stop_k8s_services.sh
