#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
OUT_DIR="uncap_data"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# Allow the HTML index so wget can parse it; accept only the 15 UFL files + the opt file.
wget -e robots=off -r -l1 -nd -np \
  -A "*.html,cap7?.txt,cap10?.txt,cap13?.txt,capa.txt,capb.txt,capc.txt,uncapopt.txt" \
  "$BASE_DIR"
