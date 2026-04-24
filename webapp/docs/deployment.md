# Deployment Guide

## System Specs

Initially the VM is setup with these specs:

| **Component**                 | **Details**                                    | **Command to Check**                                       |
| ----------------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| **OS**                        | Ubuntu 24.04.2 LTS (Noble)                     | `lsb_release -a` or `cat /etc/os-release`                  |
| **Kernel**                    | Linux 6.8.0-53-generic                         | `uname -r`                                                 |
| **Architecture**              | x86_64                                         | `lscpu`                                                    |
| **CPU**                       | 4 vCPUs (QEMU Virtual CPU version 2.5+)        | `lscpu`                                                    |
| **RAM**                       | 7.8 GiB total                                  | `free -h`                                                  |
| **Disk**                      | 98G total                                      | `df -h`                                                    |
| **Network**                   | Interface `ens18` - IP: `193.175.65.129/24`    | `ip a`                                                     |
| **Firewall**                  | Inactive                                       | `sudo ufw status`                                          |
| **Running Services**          | SSH, Fail2Ban, Postfix, Systemd-resolved, etc. | `sudo systemctl list-units --type=service --state=running` |
| **Listening Ports**           | 22 (SSH), 25 (Postfix), 53 (DNS Resolver)      | `sudo netstat -tulnp` or `ss -tulnp`                       |
| **Virtualization**            | KVM (QEMU-based)                               | `hostnamectl`                                              |
| **Docker Installed?**         | Not installed yet                              | `docker -v`                                                |
| **Docker Compose Installed?** | Not installed yet                              | `docker-compose -v`                                        |

The following Ports are open:
- `22`
- `80`
- `443`

## Setup

### Users

| **User**       | **Purpose**                       | **Login Access?** | **Group Membership** |
| -------------- | --------------------------------- | ----------------- | -------------------- |
| **`stein`**    | Personal admin account            | ✅ Yes             | `sudo`, `docker`     |
| **`deployer`** | Handles Git updates & runs Docker | ✅ Yes (SSH only)  | `docker`             |

To ensure secure deployment, we create a single unprivileged deployment user (`deployer`) to manage both Git and Docker tasks.

```bash
sudo adduser --disabled-password --gecos "" deployer
```

This user:

- Can **log in via SSH** (password authentication disabled, only key-based access).
- Will be used for **deployments and running Docker containers**.
- Does **not** have sudo access to limit security risks.

Grant `deployer` Docker access:

```bash
sudo usermod -aG docker deployer
sudo usermod -aG docker stein  # optional
```

Now `deployer` can run **Docker commands without `sudo`**.

#### Switch user

Switch to `deployer` user:

```bash
sudo -u deployer -s
```

- `-u deployer` specifies that we want to run the command as `deployer` instead of root

- `-s` starts an interactive shell for `deployer` and 

### SSH Setup

Since deployments require GitHub access, we set up **SSH authentication**.

Generate an SSH Key:

```bash
ssh-keygen -t ed25519 -C "GitHub Deploy Key for system180"
```

Press **Enter** to accept the default location (`~/.ssh/id_ed25519`).

The generated public key now needs to be added to GitHub.

Retrieve the _public key_:

```bash
cat ~/.ssh/id_ed25519.pub
```

 Copy the public key to the clip board.

#### Add Public Key to GitHub

1. Go to GitHub → Your Repo ([`system180`](https://github.com/ptrstn/system180)) → `Settings` → `Deploy keys`.
2. Click `Add deploy key`.
3. Paste the _public key_ and uncheck `Allow write access` (only read is needed).
4. Choose a Title, e.g. `GitHub Deploy Key for system180`
5. Click `Save`.

 Now `deployer` can pull updates from GitHub securely and without password authentication.

#### Clone the Repository

```bash
cd ~
git clone git@github.com:ptrstn/system180.git
```

```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Enter: `yes`

Now `deployer` owns the repo and can pull updates.

### Docker

We [install](https://docs.docker.com/engine/install/ubuntu/) Docker from the **official Docker repository** to ensure the latest version.

First, remove any existing conflicting packages:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

Add the docker repository:

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

Install docker:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify the installation:

```bash
sudo docker run hello-world
```

✅Docker is now installed.

### SSL Certificate Setup

Since the web application requires HTTPS, we use [Let’s Encrypt](https://letsencrypt.org/) to generate a free SSL certificate.

Install `certbot`

```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx -y
```

Request the SSL Certificate:

```bash
sudo certbot --nginx -d dfki-3129.dfki.de
```

Certbot will: 

- Verify the domain.
- Generate an SSL certificate.
- Automatically configure Nginx for HTTPS.

And creates the following files in `/etc/letsencrypt/live/dfki-3129.dfki.de/`:

- `cert.pem`
- `chain.pem`
- `fullchain.pem`
- `privkey.pem`

Since Let’s Encrypt files are owned by root, Nginx inside Docker may not have access.

Run:

```bash
sudo chmod -R 755 /etc/letsencrypt/live/
sudo chmod -R 755 /etc/letsencrypt/archive/
```

Now Nginx inside Docker can read the SSL certificates.

Set Up Auto-Renewal:

```bash
sudo certbot renew --dry-run
```

If this works, your certificate will **auto-renew**.

## Run web application 

The repository includes a `compose.yaml` file with FastAPI + Nginx, so we can run the app using Docker Compose.

As [`deployer`](#switch-user), navigate to the Deployment Directory:

```bash
cd /home/deployer/system180
```

Pull latest code:

```bash
git pull origin main
```

For deployment on the Ubuntu server, run `docker compose` with the `production` profile:

```bash
docker compose --profile production up -d
```

- Starts FastAPI (`webapp`) (internally exposed on `8000`)
- Nginx (`nginx` with SSL) (externally exposed on `80` and `443`)
- Uses Let's Encrypt certs from `/etc/letsencrypt/`
- Runs on `https://dfki-3129.dfki.de`

If you want to run the `webapp` without `nginx` and SSL certificates, simply omit `--profile production`

```bash
docker compose up
```

| **Environment**         | **Command**                                 | **What Happens?**                                              |
|-------------------------|---------------------------------------------|----------------------------------------------------------------|
| **Development (Local)** | `docker compose up`                         | Runs `webapp` directly → Accessible at `http://localhost:8000` |
| **Production (Server)** | `docker compose --profile production up -d` | Runs `nginx` (SSL) + `webapp` behind a reverse proxy           |
