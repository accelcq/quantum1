# This script deploys a Quantum application to IBM Cloud Code Engine without github.
# Ensure you have the IBM Cloud CLI and Code Engine plugin installed.

# contents: deploy.sh
#!/bin/bash
set -e

# Load .env-style secrets
export $(grep -v '^#' env.local | xargs)

# Use API key login (non-interactive)
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud target -g AccelCQ_Resource_Group -r "$IBM_CLOUD_REGION"

# Select or create project
PROJECT_NAME="quantum1-project"
if ! ibmcloud ce project select --name "$PROJECT_NAME" > /dev/null 2>&1; then
  ibmcloud ce project create --name "$PROJECT_NAME"
  ibmcloud ce project select --name "$PROJECT_NAME"
fi

# Create namespace if it doesn't exist
if ! ibmcloud cr namespaces | grep -qw "$IBM_CLOUD_NAMESPACE"; then
  ibmcloud cr namespace-add "$IBM_CLOUD_NAMESPACE"
fi

ibmcloud cr login

# Create or update secret in Code Engine
ibmcloud ce secret delete --name quantum1-secrets --force || true
ibmcloud ce secret create --name quantum1-secrets --from-literal IBM_QUANTUM_API_TOKEN="$IBM_QUANTUM_API_TOKEN"

docker build -t quantum1 .
docker tag quantum1 us.icr.io/$IBM_CLOUD_NAMESPACE/quantum1:latest
docker push us.icr.io/$IBM_CLOUD_NAMESPACE/quantum1:latest

ibmcloud ce application apply --file deploy.yaml

echo "🌍 Deployment complete. Access your app at:"
ibmcloud ce application get --name quantum1 --output url
