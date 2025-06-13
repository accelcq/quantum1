# Troubleshooting Guide for IBM Cloud Quantum Project Deployment

---

## Table of Contents

1. [General Troubleshooting Overview](#general-troubleshooting-overview)
2. [Authentication Issues](#authentication-issues)
3. [ImagePullBackOff / Pod Stuck Issues](#imagepullbackoff--pod-stuck-issues)
4. [IBM Cloud Container Registry (ICR) Cleanup](#ibm-cloud-container-registry-icr-cleanup)
5. [Network ACL and DNS Troubleshooting](#network-acl-and-dns-troubleshooting)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Full Script: cleanupibmclouddiskspace.sh](#full-script-cleanupibmclouddiskspacesh)
8. [Common IBM Cloud CLI Commands](#common-ibm-cloud-cli-commands)
9. [Expired Kubernetes Token (BXNIM0408E)](#expired-kubernetes-token-bxnim0408e)
10. [IBM Cloud Support](#ibm-cloud-support)

---

## General Troubleshooting Overview

Make sure you're running commands from your project root directory (e.g., `~/Projects/Qiskit/qiskit_100_py311`).

### Initial Setup Steps for First-Time Developers:

1. Ensure you're logged in to IBM Cloud:

   ```bash
   ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
   ```

2. Authenticate to Container Registry:

   ```bash
   ibmcloud cr login
   ```

3. Download Kubernetes config:

   ```bash
   ibmcloud ks cluster config --cluster quantum1-cluster
   ```

4. Test Kubernetes CLI:

   ```bash
   kubectl get nodes
   ```

> Always run these steps before any `kubectl` commands.

You can also use GitHub Codespaces or VS Code terminal to run these steps interactively.

---

## Authentication Issues

If you get an error like:

```bash
Provided refresh token is expired
```

Re-authenticate:

```bash
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud ks cluster config --cluster quantum1-cluster
kubectl get nodes
```

---

## ImagePullBackOff / Pod Stuck Issues

When the pod shows `ImagePullBackOff`, it means your image is inaccessible from the cluster.

### Solution: Create a Kubernetes image pull secret

```bash
kubectl create secret docker-registry icr-secret \
  --docker-server=us.icr.io \
  --docker-username=iamapikey \
  --docker-password="$IBM_CLOUD_API_KEY" \
  --docker-email=you@example.com \
  -n <Your IBM Container Registry namespace>

kubectl patch serviceaccount default -n <Your IBM Container Registry namespace> \
  -p '{"imagePullSecrets":[{"name":"icr-secret"}]}'
```

---

## IBM Cloud Container Registry (ICR) Cleanup

To resolve quota-related issues:

* [Quota Troubleshooting](https://cloud.ibm.com/docs/Registry?topic=Registry-troubleshoot-quota)
* [Check Quota Usage](https://cloud.ibm.com/docs/Registry?topic=Registry-registry_quota#registry_quota_get)
* [Free Up Quota](https://cloud.ibm.com/docs/Registry?topic=Registry-registry_quota#registry_quota_freeup)
* [Upgrade Plan](https://cloud.ibm.com/docs/Registry?topic=Registry-registry_overview&interface=ui#registry_plan_upgrade)

### Full Script: cleanupibmclouddiskspace.sh

```bash
#!/bin/bash
# cleanupibmclouddiskspace.sh

NAMESPACE=<Your IBM Container Registry namespace>
REGION=us.icr.io

if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
  echo "🔑 Loaded environment variables from .env.local"
else
  echo "Warning: .env.local file not found. Please ensure it exists."
fi

if [ -z "$IBM_CLOUD_API_KEY" ]; then
  echo "Error: IBM_CLOUD_API_KEY not set."
  exit 1
fi

ibmcloud login --apikey $IBM_CLOUD_API_KEY
ibmcloud target -r us-south
ibmcloud cr region-set $REGION
ibmcloud cr login

if ! command -v jq &> /dev/null; then
  echo "jq is required but not installed."
  exit 1
fi

if ! ibmcloud is target >/dev/null 2>&1; then
  echo "Not logged in to IBM Cloud."
  exit 1
fi

if ! ibmcloud cr namespace-list | grep -q "$NAMESPACE"; then
  echo "Namespace $NAMESPACE does not exist."
  exit 1
fi

images_output=$(ibmcloud cr image-digests --restrict $NAMESPACE 2>&1)
status=$?

if [ $status -ne 0 ]; then
  echo "Failed to retrieve images: $images_output"
  exit 1
fi

echo "$images_output" | awk 'NR>2 && NF>0 {print}' | while read -r line; do
  repo=$(echo "$line" | awk '{print $1}')
  digest=$(echo "$line" | awk '{print $2}')
  if [[ "$digest" == sha256:* ]]; then
    full_image="$repo@$digest"
    echo "Deleting image: $full_image"
    ibmcloud cr image-rm "$full_image"
  fi
done
```

---

## Network ACL and DNS Troubleshooting

Check:

* VPC Infrastructure > Network ACLs
* Security Groups > Ensure outbound TCP 443 is allowed

### Debug from Pod:

```bash
kubectl run debug-dns --rm -it --image=nicolaka/netshoot -- bash
nslookup registry-1.docker.io
curl https://registry-1.docker.io/v2/
```

---

## Post-Deployment Validation

After GitHub workflow or manual deployment:

```bash
kubectl get pods -n <Your IBM Container Registry namespace>
kubectl get svc -n <Your IBM Container Registry namespace>
kubectl logs -n <Your IBM Container Registry namespace> <pod-name>
```

Use port-forward to test locally:

```bash
kubectl port-forward svc/quantum1-service 8000:8000 -n <Your IBM Container Registry namespace>
```

Visit:

```http
http://localhost:8000
```

Make sure your React app and backend API respond correctly.

---

## Common IBM Cloud CLI Commands

```bash
ibmcloud cr login
ibmcloud cr namespace-add <Your IBM Container Registry namespace>
ibmcloud cr image-list
ibmcloud cr image-digests
ibmcloud cr image-rm <image@sha256:...>
ibmcloud cr trash-list
ibmcloud cr quota
```

---

## Expired Kubernetes Token (BXNIM0408E)

### Issue:

```text
Unable to connect to the server: failed to refresh token: oauth2: cannot fetch token: 400 Bad Request
```

### Fix:

```bash
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud ks cluster config --cluster quantum1-cluster
kubectl get nodes
kubectl get pods -n <Your IBM Container Registry namespace>
```

### Optional Script: refresh\_k8s\_auth.sh

```bash
#!/bin/bash
source .env.local
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
ibmcloud ks cluster config --cluster quantum1-cluster
kubectl get nodes
kubectl get pods -n <Your IBM Container Registry namespace>
```

```bash
chmod +x refresh_k8s_auth.sh
./refresh_k8s_auth.sh
```

---

## IBM Cloud Support

If none of the above resolves your issue, contact IBM Cloud Support with the error logs and detailed configuration.

👉 [IBM Cloud Support Center](https://cloud.ibm.com/unifiedsupport/supportcenter)

---

End of Guide ✅
