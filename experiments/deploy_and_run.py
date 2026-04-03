#!/usr/bin/env python3
"""Deploy and run reviewer experiments on Atlas via paramiko."""
from __future__ import annotations

import paramiko
import os
import sys

ATLAS_HOST = "100.68.134.21"
ATLAS_USER = "claude"
ATLAS_PASS = "roZes9090!~"
REMOTE_DIR = "/home/claude/turboquant-experiments"

def main():
    print("Connecting to Atlas...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ATLAS_HOST, username=ATLAS_USER, password=ATLAS_PASS)
    print("Connected.")

    # Create remote directory
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}/results")

    # Upload experiment script
    sftp = ssh.open_sftp()
    local_script = os.path.join(os.path.dirname(__file__), "run_reviewer_experiments.py")
    remote_script = f"{REMOTE_DIR}/run_reviewer_experiments.py"
    print(f"Uploading {local_script} -> {remote_script}")
    sftp.put(local_script, remote_script)
    sftp.close()

    # Check what's available on Atlas
    print("\nChecking Atlas environment...")
    for cmd in [
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader",
        "python3 -c 'import sentence_transformers; print(\"sentence-transformers:\", sentence_transformers.__version__)'",
        "python3 -c 'import sklearn; print(\"sklearn:\", sklearn.__version__)'",
        "python3 -c 'import scipy; print(\"scipy:\", scipy.__version__)'",
        "python3 -c 'import psycopg2; print(\"psycopg2:\", psycopg2.__version__)'",
        "psql -c 'SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL' atlas 2>/dev/null",
    ]:
        _, stdout, stderr = ssh.exec_command(cmd, timeout=15)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(f"  {out}")
        if err and "Warning" not in err:
            print(f"  [stderr] {err}")

    # Install missing deps if needed
    print("\nInstalling any missing packages...")
    ssh.exec_command("pip install scipy scikit-learn 2>/dev/null")

    # Start experiments in tmux (so they survive disconnection)
    # Run experiments that DON'T need GPU first (ood, retrieval, crosslingual)
    # Then E5/STS/eigenspectrum which need sentence-transformers on CPU
    print("\nStarting experiments in tmux session 'reviewer-exp'...")
    cmd = (
        f"tmux kill-session -t reviewer-exp 2>/dev/null; "
        f"tmux new-session -d -s reviewer-exp "
        f"'cd {REMOTE_DIR} && "
        f"python3 run_reviewer_experiments.py --all 2>&1 | tee experiment_log.txt'"
    )
    _, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f"tmux started: {out} {err}")

    # Verify it's running
    _, stdout, _ = ssh.exec_command("tmux ls 2>/dev/null")
    print(f"tmux sessions: {stdout.read().decode().strip()}")

    print("\nExperiments are running on Atlas in tmux session 'reviewer-exp'.")
    print(f"Monitor with: ssh claude@{ATLAS_HOST} 'tmux attach -t reviewer-exp'")
    print(f"Check results: ssh claude@{ATLAS_HOST} 'ls -la {REMOTE_DIR}/results/'")

    ssh.close()


if __name__ == "__main__":
    main()
