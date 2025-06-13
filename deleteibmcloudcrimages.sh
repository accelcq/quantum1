#!/bin/bash
# deleteibmcloudcrimages.sh
# This script deletes all images in a specified IBM Cloud Container Registry namespace.
# Ensure the script is run with bash
# Usage: bash deleteibmcloudcrimages.sh
# Check if the script is run with bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script must be run with bash. Please run 'bash deleteibmcloudcrimages.sh'."
  exit 1
fi
# Check if the required environment variables are set
# Set the namespace and region variables
# Set the namespace and region variables
NAMESPACE=quantum1space
REGION=us.icr.io
# read .env.local file
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
  echo "🔑 Loaded environment variables from .env.local"
else
  echo "Warning: .env.local file not found. Please ensure it exists with the required variables."
fi
# Check if IBM_CLOUD_API_KEY is set
if [ -z "$IBM_CLOUD_API_KEY" ]; then
  echo "Error: IBM_CLOUD_API_KEY is not set. Please set it in .env.local or export it as an environment variable."
  exit 1
fi


ibmcloud login --apikey $IBM_CLOUD_API_KEY
# Check if the required environment variables are set

ibmcloud target -r us-south
ibmcloud cr region-set $REGION
# No 'namespace-set' command; ensure namespace exists instead
ibmcloud cr login
# Ensure jq is installed for JSON parsing
if ! command -v jq &> /dev/null; then
  echo "jq is required but not installed. Please install jq to proceed."
  exit 1
fi
# Ensure ibmcloud CLI is installed and logged in
if ! command -v ibmcloud &> /dev/null; then
  echo "ibmcloud CLI is required but not installed. Please install ibmcloud CLI to proceed."
  exit 1
fi
if ! ibmcloud is target >/dev/null 2>&1; then
  echo "You must be logged in to IBM Cloud. Please run 'ibmcloud login' first."
  exit 1
fi
# Ensure the namespace exists
if ! ibmcloud cr namespace-list | grep -q "$NAMESPACE"; then
  echo "Namespace $NAMESPACE does not exist. Please create it first."
  exit 1
fi

# List all images in the namespace and delete them by digest
images_output=$(ibmcloud cr image-digests --restrict $NAMESPACE 2>&1)
status=$?

if [ $status -ne 0 ]; then
  echo "Failed to retrieve images: $images_output"
  exit 1
fi

# Skip header lines and process each image line
echo "$images_output" | awk 'NR>2 && NF>0 {print}' | while read -r line; do
  repo=$(echo "$line" | awk '{print $1}')
  digest=$(echo "$line" | awk '{print $2}')
  # Only process lines with a valid sha256 digest
  if [[ "$digest" == sha256:* ]]; then
    full_image="$repo@$digest"
    echo "Deleting image: $full_image"
    ibmcloud cr image-rm "$full_image"
  fi
done
