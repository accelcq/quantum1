# quantum1.yaml for IBM Cloud Kubernetes

---
apiVersion: v1
kind: Secret
metadata:
  name: quantum1-secrets
  namespace: default
stringData:
  IBM_QUANTUM_API_TOKEN: "<paste-your-quantum-token-here>"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum1-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: quantum1
  template:
    metadata:
      labels:
        app: quantum1
    spec:
      containers:
      - name: quantum1
        image: us.icr.io/accelcq/quantum1:latest
        ports:
        - containerPort: 8000
        env:
        - name: IBM_QUANTUM_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantum1-secrets
              key: IBM_QUANTUM_API_TOKEN
        volumeMounts:
        - mountPath: /app/execution_log.log
          name: log-volume
          subPath: execution_log.log
      volumes:
      - name: log-volume
        persistentVolumeClaim:
          claimName: quantum1-log-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: quantum1-log-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi

---
apiVersion: v1
kind: Service
metadata:
  name: quantum1-service
spec:
  type: LoadBalancer
  selector:
    app: quantum1
  ports:
    - port: 80
      targetPort: 8000


# Terraform Plan for IBM VPC Instance

provider "ibm" {
  ibmcloud_api_key = var.ibmcloud_api_key
  region           = var.region
}

resource "ibm_is_vpc" "quantum_vpc" {
  name = "quantum-vpc"
}

resource "ibm_is_subnet" "quantum_subnet" {
  name           = "quantum-subnet"
  vpc            = ibm_is_vpc.quantum_vpc.id
  total_ipv4_address_count = 256
  zone           = var.zone
}

resource "ibm_is_instance" "quantum_vm" {
  name           = "quantum-vm"
  image          = var.image_id
  profile        = "bx2-2x8"
  vpc            = ibm_is_vpc.quantum_vpc.id
  primary_network_interface {
    subnet = ibm_is_subnet.quantum_subnet.id
  }
  zone = var.zone
  keys = [var.ssh_key_id]
}


# Ansible Playbook for Docker + App on VM

---
- name: Setup quantum1 app on IBM Cloud VM
  hosts: quantum
  become: yes
  vars:
    docker_image: "us.icr.io/accelcq/quantum1:latest"
  tasks:
    - name: Install Docker
      apt:
        name: docker.io
        state: present
        update_cache: yes

    - name: Ensure Docker service is running
      service:
        name: docker
        state: started
        enabled: true

    - name: Pull Docker image
      command: docker pull {{ docker_image }}

    - name: Run Docker container
      command: docker run -d -p 8000:8000 {{ docker_image }}


# Comparison: Terraform vs Ansible

| Feature                        | Terraform                         | Ansible                           |
|-------------------------------|------------------------------------|------------------------------------|
| **Purpose**                   | Infrastructure provisioning        | Server configuration               |
| **Stateful**                  | Yes                                | No                                 |
| **Declarative or Procedural** | Declarative                        | Procedural                         |
| **Best For**                  | Creating VPC, VM, networks         | Installing packages, running apps |
| **Execution Order**           | Parallel by design                 | Step-by-step                       |
| **Preferred Use**             | First to create VM infra           | Then configure using Ansible       |

---

Combine both by running Terraform first to create VM + SSH, and then run Ansible to configure the VM.
