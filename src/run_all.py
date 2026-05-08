"""
src/run_all.py
──────────────
Single command to run the full FL experiment.

Fixes applied:
  1. Downloads dataset ONCE (in the launcher) before launching anything.
  2. Starts server as a subprocess, then polls port 8080 until it is
     actually accepting connections before spawning clients.
  3. All processes use the same absolute data root path.

Usage (run from the repo root):
    python src/run_all.py --config configs/fedavg_baseline.yaml
    python src/run_all.py --config configs/fedmia_cifar100.yaml
    python src/run_all.py --config configs/fedmia_mnist.yaml
"""

import argparse
import os
import sys
import socket
import subprocess
import time
import yaml

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_ADDR = f"{SERVER_HOST}:{SERVER_PORT}"
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SRC_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pre_download_dataset(cfg: dict):
    """
    Download the dataset exactly once inside the launcher process.
    All subprocesses will then find the files already on disk.
    """
    import torchvision
    ds   = cfg["data"]["dataset"].upper()
    root = os.path.join(REPO_ROOT, "data")
    os.makedirs(root, exist_ok=True)

    print(f"[Launcher] Checking / downloading {ds} to {root} …")
    try:
        if ds == "MNIST":
            torchvision.datasets.MNIST(root, train=True,  download=True)
            torchvision.datasets.MNIST(root, train=False, download=True)
        elif ds == "FMNIST":
            torchvision.datasets.FashionMNIST(root, train=True,  download=True)
            torchvision.datasets.FashionMNIST(root, train=False, download=True)
        elif ds == "CIFAR10":
            torchvision.datasets.CIFAR10(root, train=True,  download=True)
            torchvision.datasets.CIFAR10(root, train=False, download=True)
        elif ds == "CIFAR100":
            torchvision.datasets.CIFAR100(root, train=True,  download=True)
            torchvision.datasets.CIFAR100(root, train=False, download=True)
        else:
            print(f"[Launcher] Unknown dataset {ds}, skipping pre-download.")
    except Exception as e:
        print(f"[Launcher] Pre-download warning: {e}")
    print(f"[Launcher] Dataset {ds} ready.\n")


def wait_for_server(host: str, port: int, timeout: float = 120.0, interval: float = 1.0):
    """
    Block until the server is accepting TCP connections on host:port,
    or raise TimeoutError after `timeout` seconds.
    """
    deadline = time.time() + timeout
    print(f"[Launcher] Waiting for server at {host}:{port} ", end="", flush=True)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(" ready!", flush=True)
                return
        except (ConnectionRefusedError, OSError):
            print(".", end="", flush=True)
            time.sleep(interval)
    raise TimeoutError(f"Server at {host}:{port} did not start within {timeout}s")


def spawn(cmd: list, label: str) -> subprocess.Popen:
    """Spawn a subprocess, inheriting stdout/stderr."""
    print(f"[Launcher] Starting {label}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": SRC_DIR},   # ensure src/ is on path
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch full FL experiment")
    parser.add_argument("--config", type=str, default="configs/fedavg_baseline.yaml")
    args = parser.parse_args()

    # Resolve config path relative to repo root
    config_path = args.config if os.path.isabs(args.config) else \
                  os.path.join(REPO_ROOT, args.config)

    cfg         = load_cfg(config_path)
    num_clients = cfg["federated"]["num_clients"]

    print("=" * 60)
    print(f" FL Experiment : {cfg['experiment']['name']}")
    print(f" Dataset       : {cfg['data']['dataset']}")
    print(f" Clients       : {num_clients}")
    print(f" Rounds        : {cfg['federated']['num_rounds']}")
    print(f" Server        : {SERVER_ADDR}")
    print("=" * 60 + "\n")

    # ── Step 1: download dataset once ────────────────────────────────────────
    pre_download_dataset(cfg)

    # ── Step 2: start server subprocess ──────────────────────────────────────
    server_proc = spawn(
        [sys.executable, os.path.join(SRC_DIR, "server.py"),
         "--config", config_path,
         "--server", f"0.0.0.0:{SERVER_PORT}"],
        label="Server",
    )

    # ── Step 3: wait until server is actually ready ───────────────────────────
    try:
        wait_for_server(SERVER_HOST, SERVER_PORT, timeout=120.0)
    except TimeoutError as e:
        print(f"\n[Launcher] ERROR: {e}")
        server_proc.terminate()
        sys.exit(1)

    # Give gRPC one extra second to fully initialise
    time.sleep(2.0)

    # ── Step 4: spawn all client subprocesses ─────────────────────────────────
    client_procs = []
    for cid in range(num_clients):
        p = spawn(
            [sys.executable, os.path.join(SRC_DIR, "client.py"),
             "--cid",    str(cid),
             "--config", config_path,
             "--server", SERVER_ADDR],
            label=f"Client {cid}",
        )
        client_procs.append(p)
        time.sleep(0.3)   # small stagger to avoid hammering gRPC at once

    print(f"\n[Launcher] All {num_clients} clients launched. Waiting for FL to finish …\n")

    # ── Step 5: wait for clients ──────────────────────────────────────────────
    for cid, p in enumerate(client_procs):
        p.wait()
        status = "OK" if p.returncode == 0 else f"ERROR (code {p.returncode})"
        print(f"[Launcher] Client {cid} finished — {status}")

    # ── Step 6: wait for server ───────────────────────────────────────────────
    server_proc.wait()
    print(f"\n[Launcher] Server finished — exit code {server_proc.returncode}")
    print("[Launcher] Experiment complete. Check results/ for outputs.")


if __name__ == "__main__":
    main()
