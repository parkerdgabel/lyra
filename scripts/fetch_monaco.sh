#!/usr/bin/env bash
set -euo pipefail

# Vendors Monaco editor (minified AMD build) into crates/lyra-notebook-app/ui/vs
# Requires: curl, tar
# Version can be adjusted as needed

VERSION="0.45.0"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT_DIR/crates/lyra-notebook-app/ui/vs"

echo "Vendoring monaco-editor@$VERSION into $DEST_DIR"
mkdir -p "$DEST_DIR"

TMPDIR="$(mktemp -d)"
TARBALL_URL="https://registry.npmjs.org/monaco-editor/-/monaco-editor-$VERSION.tgz"

echo "Downloading $TARBALL_URL ..."
curl -L "$TARBALL_URL" -o "$TMPDIR/monaco.tgz"

echo "Extracting min/vs ..."
mkdir -p "$TMPDIR/p"
tar -xzf "$TMPDIR/monaco.tgz" -C "$TMPDIR/p"

# The tarball contains package/min/vs
if [ ! -d "$TMPDIR/p/package/min/vs" ]; then
  echo "Error: expected path package/min/vs not found in tarball" >&2
  exit 1
fi

rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"
cp -R "$TMPDIR/p/package/min/vs/"* "$DEST_DIR/"

echo "Done. Local loader at ui/vs/loader.js"

