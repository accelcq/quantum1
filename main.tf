# main.tf 
# to deploy dockerize image using terraform script
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

