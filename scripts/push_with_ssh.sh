#!/usr/bin/env bash
# push_with_ssh.sh
# Usage: ./push_with_ssh.sh [repo_dir] [branch]
# If repo_dir is omitted, uses current directory. If branch is omitted, uses current branch or 'main'.

set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
BRANCH="${2:-main}"
PRINT_PUB="${PRINT_PUB:-0}"

echo "Repository: $REPO_DIR"
cd "$REPO_DIR"

# start ssh agent if not already
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  echo "Starting ssh-agent..."
  eval "$(ssh-agent -s)"
else
  echo "ssh-agent already running"
fi

# add private key
KEY_PATH="$HOME/.ssh/github"
if [ -f "$KEY_PATH" ]; then
  ssh-add "$KEY_PATH" || true
  echo "Added $KEY_PATH to agent"
else
  echo "Warning: key $KEY_PATH not found"
fi

# optionally print the public key for adding to GitHub
if [ "$PRINT_PUB" = "1" ]; then
  if [ -f "$KEY_PATH.pub" ]; then
    echo "--- public key (copy this into GitHub) ---"
    cat "$KEY_PATH.pub"
    echo "--- end public key ---"
  else
    echo "Public key $KEY_PATH.pub not found"
  fi
fi

# test the connection
echo "Testing SSH connection to GitHub (this may print a welcome message)..."
ssh -T git@github.com || true

# finally push
echo "Pushing to origin/$BRANCH from $REPO_DIR"
git push origin "$BRANCH"

echo "Done."
