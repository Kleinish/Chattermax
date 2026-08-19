#!/usr/bin/env bash
set -euo pipefail

APP_SOURCE=/opt/chatterbox-piper
WORKSPACE=${WORKSPACE_ROOT:-/workspace}

mkdir -p "$WORKSPACE"

# In a normal Git/Compose checkout, the project is bind-mounted and should be
# left alone. In a packaged Unraid/published-image deployment, /workspace is a
# persistent data volume and application code is supplied by the image.
if [[ ! -f "$WORKSPACE/compose.yaml" ]]; then
  rm -rf "$WORKSPACE/web" "$WORKSPACE/scripts"
  cp -a "$APP_SOURCE/web" "$WORKSPACE/web"
  cp -a "$APP_SOURCE/scripts" "$WORKSPACE/scripts"
fi

if [[ ! -d "$WORKSPACE/corpus_sources" ]]; then
  cp -a "$APP_SOURCE/corpus_sources" "$WORKSPACE/corpus_sources"
fi

if [[ ! -f "$WORKSPACE/.env" ]]; then
  cp "$APP_SOURCE/.env.example" "$WORKSPACE/.env"
fi

for dir in input logs output rejected reports training dataset dataset_raw review; do
  mkdir -p "$WORKSPACE/$dir"
done

cd "$WORKSPACE"
exec "$@"
