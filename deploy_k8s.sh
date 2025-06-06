#deploy_k8s.sh
#!/bin/bash
set -e

# Load .env-style secrets if available
if [ -f .env.local ]; then
  export $(grep -v '^#' .env.local | xargs)
fi

# IBM Cloud login and registry setup
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud cr login

# Build and push Docker image
IMAGE_NAME="us.icr.io/$IBM_CLOUD_NAMESPACE/quantum1:latest"
docker build -t quantum1 .
docker tag quantum1 $IMAGE_NAME
docker push $IMAGE_NAME

# Kubernetes cluster config
ibmcloud ks cluster config --cluster "$K8S_CLUSTER_NAME"

# Apply K8S deployment
kubectl apply -f quantum1.yaml
kubectl rollout status deployment/quantum1-deployment
kubectl get svc quantum1-service
# Get the external IP of the service
EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
  EXTERNAL_IP=$(kubectl get svc quantum1-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  if [ -z "$EXTERNAL_IP" ]; then
    echo "Waiting for external IP..."
    sleep 10
  fi
done
echo "Quantum1 service is available at http://$EXTERNAL_IP:8080"
# Output the service URL
echo "Quantum1 service URL: http://$EXTERNAL_IP:8080"
# Output the service URL for the user
echo "You can access the Quantum1 service at: http://$EXTERNAL_IP:8080"
