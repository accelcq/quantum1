# 🎢 IBM Cloud Quantum Deployment Troubleshooting Guide

---

## Summary of Identified Issue

**✅ Pod Scheduling Works**

* Pods are created and scheduled onto nodes without issue.

**❌ Image Pulling Fails**

* External image registries such as Docker Hub and Google Container Registry (GCR) consistently timeout.
* Even basic image pulls (e.g., `k8s.gcr.io/pause:3.9`) fail.

### Final Diagnosis

Worker nodes in the IBM Cloud Kubernetes cluster are blocked from reaching external container registries (e.g., Docker Hub, GCR).

This strongly suggests an **outbound network restriction** at the VPC, Network ACL, or IBM Cloud infrastructure firewall level.

---

## 🚧 Immediate Resolution Plan

### 1. ✅ Check IBM Cloud Network Security Groups (NSGs) / Firewalls

#### A. In IBM Cloud Console

* Go to: **Menu → VPC Infrastructure → Network → Security groups**
* Select your cluster’s region.
* Check outbound rules:

  * ✅ Ensure **TCP on port 443** to **0.0.0.0/0** is allowed.

#### B. In CLI

```bash
ibmcloud login --sso
ibmcloud plugin install vpc-infrastructure
ibmcloud target -r <region>
ibmcloud is security-groups
ibmcloud is security-group <security-group-id>
ibmcloud is security-group-rule-add <security-group-id> outbound tcp --port-min 443 --port-max 443 --remote 0.0.0.0/0
```

### 2. 🌐 Check IBM VPC Network ACLs

* Go to: **VPC Infrastructure → Network → Network ACLs**
* Select the ACL for your worker nodes’ subnet.
* Ensure there’s an **Outbound TCP rule allowing port 443** to `0.0.0.0/0`.

### 3. 🧪 Test Node Connectivity from Inside the Cluster

```bash
kubectl run net-debug --rm -it --image=nicolaka/netshoot -- bash
```

Inside the pod:

```bash
nslookup registry-1.docker.io
curl https://registry-1.docker.io/v2/
```

Expected: JSON error confirming access (e.g., UNAUTHORIZED).

---

## ✅ IBM Container Registry Workaround (Recommended)

### Step-by-Step

```bash
ibmcloud cr login
ibmcloud cr namespace-add quantum1space

# Tag & Push
docker tag quantum1:latest us.icr.io/quantum1space/quantum1:latest
docker push us.icr.io/quantum1space/quantum1:latest
```

### Update Kubernetes YAML

```yaml
containers:
  - name: quantum1
    image: us.icr.io/quantum1space/quantum1:latest
```

### Apply to Correct Namespace

```bash
kubectl apply -f quantum1_k8s.yaml -n quantum1space
```

---

## 🔐 Fixing 401 Unauthorized (Private Image Access)

1. **Generate an IBM API Key** from [IBM Cloud Console](https://cloud.ibm.com/iam/apikeys)
2. **Create Image Pull Secret**

```bash
kubectl create secret docker-registry icr-secret \
  --docker-server=us.icr.io \
  --docker-username=iamapikey \
  --docker-password="<your-api-key>" \
  --docker-email=you@example.com \
  -n quantum1space
```

3. **Patch Service Account**

```bash
kubectl patch serviceaccount default -n quantum1space \
  -p '{"imagePullSecrets":[{"name":"icr-secret"}]}'
```

4. **Restart Pods**

```bash
kubectl rollout restart deployment quantum1-deployment -n quantum1space
```

---

## 📋 Useful IBM Cloud CLI Commands

```bash
ibmcloud cr login                           # Login to IBM CR
ibmcloud cr namespace-add quantum1space    # Add CR namespace
ibmcloud cr image-list                     # List tagged images
ibmcloud cr image-digests                  # Show all digests
ibmcloud cr image-rm <image@sha256:...>    # Delete by digest
ibmcloud cr trash-list                     # See trash images
ibmcloud cr quota                          # View free-tier limits
```

---

## 🚼 IBM CR Cleanup Tips (Free Tier Limit)

```bash
# Delete image by digest
ibmcloud cr image-rm us.icr.io/quantum1space/quantum1@sha256:<digest>

# Check if trash is empty
ibmcloud cr image-digests
```

---

## ✅ Post-Deployment Validation

```bash
kubectl get pods -n quantum1space           # Check pod status
kubectl describe pod <pod> -n quantum1space # See why it's stuck (if any)
kubectl get svc -n quantum1space            # Check service IP/port
kubectl logs -n quantum1space <pod>         # View application logs
kubectl port-forward svc/quantum1-service 8000:8000 -n quantum1space
```

Visit: [http://localhost:8000](http://localhost:8000) to test app locally.

---

## 🎯 Refresh Expired Kubernetes Token (BXNIM0408E Error)

### Issue:

```text
Unable to connect to the server: failed to refresh token: oauth2: cannot fetch token: 400 Bad Request
Response: {"errorCode":"BXNIM0408E","errorMessage":"Provided refresh token is expired"}
```

### Fix:

1. **Re-authenticate with IBM Cloud**

```bash
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"
```

2. **Reconnect to Kubernetes Cluster**

```bash
ibmcloud ks cluster config --cluster quantum1-cluster
```

3. **Verify Connection**

```bash
kubectl get nodes
kubectl get pods -n quantum1space
```

### Optional Script (refresh\_k8s\_auth.sh):

```bash
#!/bin/bash

# Load from .env.local if needed
source .env.local

# Re-authenticate
ibmcloud login --apikey "$IBM_CLOUD_API_KEY" -r "$IBM_CLOUD_REGION"

# Refresh Kube config
ibmcloud ks cluster config --cluster quantum1-cluster

# Confirm access
kubectl get nodes
kubectl get pods -n quantum1space
```

Make it executable:

```bash
chmod +x refresh_k8s_auth.sh
./refresh_k8s_auth.sh
```

---

## 🌟 Final Recommendation

If outbound network still fails after confirming ACL and NSG rules, escalate to IBM Cloud Support:

> "My Kubernetes worker nodes are unable to pull images from external registries (Docker Hub, GCR) despite VPC ACL and NSG allowing outbound HTTPS."
