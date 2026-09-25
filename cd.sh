#!/usr/bin/env bash
# Server setup crib sheet.
#
# NO CREDENTIALS IN THIS FILE. This repository is public -- anything written
# here is readable by anyone, forever, including in git history. Keep secrets
# in ~/.ssh/config, a password manager, or an untracked local file.
#
# Set up key-based SSH once and you never type a password again:
#     ssh-keygen -t ed25519 -C "btp"
#     ssh-copy-id $SERVER_USER@$SERVER_HOST
# then add to ~/.ssh/config (NOT tracked by git):
#     Host btp
#         HostName <server ip>
#         User <your user>
#         IdentityFile ~/.ssh/id_ed25519
# after which `ssh btp` just works.

SERVER_USER="${SERVER_USER:-<your-user>}"
SERVER_HOST="${SERVER_HOST:-<server-ip>}"
PROJECT_PARENT="/home/${SERVER_USER}/GOT_Compression/IITD_GOT/Prasoon"

# --- connect ---------------------------------------------------------
#     ssh "$SERVER_USER@$SERVER_HOST"        # or: ssh btp

# --- IITD network proxy ----------------------------------------------
# Needed for outbound internet (pip, HuggingFace downloads). Run in a
# SEPARATE terminal and leave it open. It will prompt for its password --
# do not store that here.
#     cd ~/ && python3 iitdproxy.py proxyAuth.txt

# --- fresh clone -----------------------------------------------------
# Destructive: wipes local changes and results on the server. Prefer
# `git pull` unless you really want a clean slate.
clone_fresh () {
    cd "$PROJECT_PARENT" || return 1
    rm -rf Graph_of_Thought
    git clone https://github.com/TREX4096/Graph_of_Thought
    cd Graph_of_Thought || return 1
}

# --- normal update ---------------------------------------------------
update () {
    cd "$PROJECT_PARENT/Graph_of_Thought" || return 1
    git pull
}

# --- environment -----------------------------------------------------
#     conda activate GOTComp-cu121
#     pip install -e ".[hpc]"
#     python scripts/generate_data.py --out data --seed 42 --n-samples 100 --small

# --- run -------------------------------------------------------------
#     pytest -q                       # expect 60 passed
#     nvidia-smi                      # pick a free GPU
#     bash scripts/run_tmux.sh        # full matrix, survives SSH drops
#     tmux attach -t got
