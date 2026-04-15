#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Script lives in checkpoints folder, so target is this folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}"

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"

FILES=(
  "sam2_hiera_tiny.pt"
  #"sam2_hiera_small.pt"
  #"sam2_hiera_base_plus.pt"
  #"sam2_hiera_large.pt"
)

mkdir -p "${CHECKPOINT_DIR}"

echo "Checkpoint target folder: ${CHECKPOINT_DIR}"

for file in "${FILES[@]}"; do
  target="${CHECKPOINT_DIR}/${file}"
  url="${BASE_URL}/${file}"

  if [[ -f "${target}" ]]; then
    echo "Skipping ${file} (already exists)."
    continue
  fi

  echo "Downloading ${file}..."
  wget -O "${target}" "${url}" || {
    echo "Failed to download ${file} from ${url}"
    exit 1
  }
done

echo "All requested checkpoints are present in ${CHECKPOINT_DIR}."
