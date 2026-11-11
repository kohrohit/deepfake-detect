#!/usr/bin/env bash
# Deploy the dfd detection service to a remote host over SSH.
#
# This is ops/install.sh's remote sibling: install.sh assumes you are already
# on the machine with the repo checked out and dfd importable, which is true
# on a laptop and false on a fresh EC2 instance. This script does the three
# things that stand between those two states -- get the code there, build an
# interpreter that can import it, and hand over to install.sh -- and it does
# them idempotently, so re-running it after a code change is the deploy.
#
# WHAT IS DELIBERATELY NOT SHIPPED. assets/models/dima806 is 328MB and is NOT
# wired into the registry: it measured at chance (AUC 0.5214, mean P(fake)
# 0.961 on reals vs 0.952 on fakes, see bench/evidence_card.json). Shipping it
# would be transfer spent on a detector that contributes nothing.
#
# WHAT IS SHIPPED, AND WHY THE ARITHMETIC ABOVE CHANGED. Slot D (gend_clip)
# cannot score without its CLIP ViT-L/14 vision tower, so
# assets/models/clip_vit_l14_visual.pt (1.2GB) IS shipped -- it is not an
# optional extra like dima806, it is the detector. That inverts the sentence
# this block used to carry: dima806 was called out as 89% of the transfer, and
# with the encoder present it is closer to 10% of it. Both facts are true; the
# tree simply grew a 1.2GB runtime asset. The small weights still matter:
# npr.pt (2.8K), gend_clip.pt (8.4K) and calibration.json are the fitted
# parts, and every one of them is tiny.
#
# PREFER --fetch-remote ON A FRESH INSTANCE. Pushing 1.2GB (plus the 890MB
# OpenAI archive, if it is in the tree) up a home uplink is the slowest part
# of this script by a wide margin, and the instance can fetch the same bytes
# from OpenAI's CDN in a fraction of the time. --fetch-remote excludes both
# files from the rsync and runs ops/fetch-assets.sh + training/convert_clip.py
# ON THE INSTANCE instead. The default stays rsync, for an instance with no
# outbound access -- there the slow uplink is the only route there is.
#
# BINDING. The service has NO authentication (see dfd.service.api), so this
# binds 127.0.0.1 on the remote host and nothing is exposed to the internet.
# Reach the dashboard with an SSH tunnel:
#     ssh -i KEY -L 8077:127.0.0.1:8077 ubuntu@HOST
#     then open http://127.0.0.1:8077
# Do not pass --allow-remote-access without an authenticating proxy in front.
#
# Usage:
#   ./ops/deploy-ec2.sh --host HOST --key KEY.pem [--user ubuntu] [--dry-run]
#                       [--fetch-remote]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="ubuntu"
HOST=""
KEY=""
DRY=0
FETCH_REMOTE=0
REMOTE_DIR="dfd"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --key)  KEY="$2";  shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --dir)  REMOTE_DIR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --fetch-remote) FETCH_REMOTE=1; shift ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$HOST" ]] || { echo "--host is required" >&2; exit 2; }
[[ -n "$KEY"  ]] || { echo "--key is required"  >&2; exit 2; }
[[ -f "$KEY"  ]] || { echo "key not found: $KEY" >&2; exit 2; }

# ssh refuses a key the group or world can read, and the failure message
# ("UNPROTECTED PRIVATE KEY FILE") is clearer than anything we would print.
chmod 600 "$KEY"

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
TARGET="$USER_NAME@$HOST"

echo "repository: $REPO"
echo "target:     $TARGET:~/$REMOTE_DIR"
echo

# Fail here rather than halfway through an rsync. A timeout on 22 is almost
# always the security group, not the key, so say so: the two have completely
# different fixes and the ssh error does not distinguish them.
echo "==> checking SSH reachability"
if ! "${SSH[@]}" -o BatchMode=yes "$TARGET" true 2>/tmp/dfd-ssh-err; then
  echo "cannot reach $TARGET over SSH:" >&2
  sed 's/^/    /' /tmp/dfd-ssh-err >&2
  echo >&2
  echo "If that is a timeout, the host is not refusing the key -- nothing is" >&2
  echo "answering on port 22. Check, in this order:" >&2
  echo "  1. the instance is running   (EC2 console, or: aws ec2 describe-instances)" >&2
  echo "  2. its security group allows inbound TCP 22 from your address" >&2
  echo "     your address: $(curl -fsS --max-time 10 https://checkip.amazonaws.com 2>/dev/null || echo '<could not determine>')" >&2
  exit 1
fi
echo "    ok"

if [[ $DRY -eq 1 ]]; then
  echo "==> dry run: would sync and install, stopping here"
  exit 0
fi

echo "==> ensuring rsync on the remote"
"${SSH[@]}" "$TARGET" 'command -v rsync >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq rsync; }'

echo "==> syncing code and weights"
# --delete keeps the remote tree honest about removed files, but the excludes
# below are NOT synced, so they must never be deleted either: --delete-excluded
# is deliberately absent. The remote venv lives in the repo dir and would be
# the first casualty.
# On the --fetch-remote route the two CLIP files are excluded here and built
# on the instance below. They are excluded rather than deleted: --delete is on,
# but rsync never deletes what it was told not to look at (--delete-excluded is
# deliberately absent), so an encoder already on the instance survives.
ENCODER_EXCLUDES=()
if [[ $FETCH_REMOTE -eq 1 ]]; then
  ENCODER_EXCLUDES=(
    --exclude='assets/models/ViT-L-14.pt'
    --exclude='assets/models/clip_vit_l14_visual.pt'
  )
  echo "    --fetch-remote: not sending the CLIP encoder; the instance will fetch it"
fi

rsync -az --delete --info=stats1 \
  -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  "${ENCODER_EXCLUDES[@]+"${ENCODER_EXCLUDES[@]}"}" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.coverage' \
  --exclude='*.pdf' \
  --exclude='*.pem' \
  --exclude='assets/models/dima806/' \
  "$REPO/" "$TARGET:$REMOTE_DIR/"

echo "==> installing interpreter and dependencies"
"${SSH[@]}" "$TARGET" "REMOTE_DIR=$REMOTE_DIR bash -s" <<'REMOTE_EOF'
set -euo pipefail
cd "$HOME/$REMOTE_DIR"

# python3-venv is not in the Ubuntu server image by default, and `python3 -m
# venv` fails with an ensurepip message rather than a missing-package one.
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  echo "    installing python3-venv"
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "    creating .venv"
  python3 -m venv .venv
fi

# The CPU wheel: the default index resolves torch to the CUDA build, ~2.5GB of
# nvidia-* wheels this instance has no GPU to use. This pipeline is CPU-only
# by design.
echo "    installing dfd (CPU torch)"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -e .

.venv/bin/python -c "import dfd, torch, cv2; print('    dfd ok, torch', torch.__version__, 'cv2', cv2.__version__)"
REMOTE_EOF

if [[ $FETCH_REMOTE -eq 1 ]]; then
  echo "==> fetching and converting the CLIP encoder on the instance"
  # Both steps are skipped when their output is already present, so re-running
  # the deploy does not re-download 890MB. convert_clip --verify re-checks the
  # archive digest and the converted tower's parameter count against the
  # architecture constants, so a truncated download fails here rather than as
  # a disappointing AUC three hours later.
  "${SSH[@]}" "$TARGET" "REMOTE_DIR=$REMOTE_DIR bash -s" <<'FETCH_EOF'
set -euo pipefail
cd "$HOME/$REMOTE_DIR"
if [[ -f assets/models/clip_vit_l14_visual.pt ]]; then
  echo "    encoder already present, skipping fetch and convert"
else
  ./ops/fetch-assets.sh
  PYTHONPATH=src:. .venv/bin/python -m training.convert_clip \
    --archive assets/models/ViT-L-14.pt \
    --out assets/models/clip_vit_l14_visual.pt --verify
fi
FETCH_EOF
fi

echo "==> installing the service"
"${SSH[@]}" "$TARGET" "cd \$HOME/$REMOTE_DIR && PYTHON=\$HOME/$REMOTE_DIR/.venv/bin/python ./ops/install.sh"

echo "==> restarting the service onto the synced code"
# `ops/install.sh` ends in `systemctl --user enable --now`, which STARTS a
# stopped unit and does nothing at all to a running one. So without this,
# a second deploy syncs new code to disk and leaves the old process serving
# it -- the deploy "succeeds", the tree is current, and the running service
# is whatever it was before. Measured 2026-11-10: a deploy onto a service
# already up for 11.7h left the uptime at 11.7h. This script's own header
# claims re-running it after a code change IS the deploy; this is the line
# that makes that true.
"${SSH[@]}" "$TARGET" 'systemctl --user restart dfd.service && sleep 2 && systemctl --user is-active dfd.service'

echo "==> enabling linger (survive logout and reboot)"
# install.sh only PRINTS this advice, which is right for a laptop with a login
# session and wrong for a server: a systemd USER unit with no lingering is
# killed the moment the deploying SSH connection closes, so the deploy would
# "succeed" and leave nothing running.
"${SSH[@]}" "$TARGET" 'sudo loginctl enable-linger "$USER" && loginctl show-user "$USER" | grep -q "Linger=yes" && echo "    linger enabled"'

echo "==> verifying"
"${SSH[@]}" "$TARGET" "curl -fsS --max-time 10 -o /dev/null -w '    dashboard HTTP %{http_code}\n' http://127.0.0.1:8077/ || echo '    dashboard did not answer'"

cat <<TUNNEL

Deployed. The service binds 127.0.0.1 on the remote host and has no
authentication, so reach it through a tunnel:

    ssh -i $KEY -L 8077:127.0.0.1:8077 $TARGET
    open http://127.0.0.1:8077

Score a file without the dashboard:

    ssh -i $KEY $TARGET 'cd $REMOTE_DIR && .venv/bin/python -c "
from dfd.cli import main; import sys
sys.exit(main([\"score\",\"IMAGE.jpg\",
  \"--calibration\",\"assets/models/calibration.json\",
  \"--face-model\",\"assets/models/face_detection_yunet_2023mar.onnx\",
  \"--pretty\"]))"'

Or drop files in the watched inbox:  ~/.local/share/dfd/inbox
TUNNEL
