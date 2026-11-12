#!/usr/bin/env bash
# Fetch the third-party model weights this repo needs and verify them.
#
# Every weight file is gitignored (*.onnx, and assets/models/ generally), so a
# fresh checkout has none and the whole pipeline abstains with
# `weights_absent`. That is correct behaviour and a terrible deployment story:
# until 2026-09-23 the only record of WHERE these files come from was a URL in
# assets/manifest.yaml and whatever the last person to set up a machine
# remembered. This script is that record, executable.
#
# WHY THE HASHES ARE PINNED. The asset gate (dfd/asset_scan.py) checks that a
# file on disk is REGISTERED and commercially cleared. Registration is a claim
# about a logical asset id, not about bytes: a file swapped at the same path
# passes the gate unchanged. Pinning here is what makes the manifest's licence
# claim attach to specific content — and for the embedder it is stronger than
# housekeeping, because a different recogniser at that path silently changes
# what every identity certificate in the benchmark means.
#
# Usage:
#   ./ops/fetch-assets.sh            # fetch what is missing, verify everything
#   ./ops/fetch-assets.sh --force    # re-fetch even if present
#   ./ops/fetch-assets.sh --check    # verify only; fetch nothing; non-zero on drift
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/assets/models"

# id|path|sha256|url
# GitHub serves LFS-tracked files as pointer stubs from raw.githubusercontent;
# media.githubusercontent.com/media is the endpoint that returns the bytes.
# Fetching the wrong one yields a 133-byte "file" that OpenCV rejects with an
# ONNX parse error several layers away from the cause.
ASSETS=(
  "yunet_face_detector|face_detection_yunet_2023mar.onnx|8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4|https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  "sface_face_recogniser|face_recognition_sface_2021dec.onnx|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79|https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

mode="fetch"
case "${1:-}" in
  --force) mode="force" ;;
  --check) mode="check" ;;
  "") ;;
  *) echo "usage: $0 [--force|--check]" >&2; exit 2 ;;
esac

mkdir -p "$DEST"
status=0

for entry in "${ASSETS[@]}"; do
  IFS='|' read -r id name want url <<< "$entry"
  path="$DEST/$name"

  if [[ -f "$path" && "$mode" != "force" ]]; then
    :
  elif [[ "$mode" == "check" ]]; then
    echo "MISSING  $id ($name)"
    status=1
    continue
  else
    echo "fetching $id ..."
    tmp="$path.partial"
    # Download to a temporary name and move only after the hash matches, so an
    # interrupted fetch never leaves a truncated file that later looks present.
    if ! curl -fSL --retry 3 --connect-timeout 20 -o "$tmp" "$url"; then
      echo "FAILED   $id: download error from $url" >&2
      rm -f "$tmp"
      status=1
      continue
    fi
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    if [[ "$got" != "$want" ]]; then
      echo "FAILED   $id: sha256 mismatch" >&2
      echo "         want $want" >&2
      echo "         got  $got" >&2
      echo "         (a 133-byte file here means the LFS pointer was fetched, not the object)" >&2
      rm -f "$tmp"
      status=1
      continue
    fi
    mv "$tmp" "$path"
  fi

  got="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$got" == "$want" ]]; then
    echo "OK       $id  $name"
  else
    echo "DRIFT    $id: $name on disk does not match the pinned hash" >&2
    echo "         want $want" >&2
    echo "         got  $got" >&2
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo
  echo "All pinned assets present and verified in $DEST"
  echo "Licences: assets/manifest.yaml (both are permissive and commercially cleared)"
else
  echo
  echo "One or more assets are missing or do not match. The pipeline will abstain" >&2
  echo "with weights_absent rather than guess, which is safe but detects nothing." >&2
fi
exit $status
