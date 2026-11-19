#!/usr/bin/env bash
# Build the documentation site, with the WASM notebook under site/app/.
# Usage: scripts/build_site.sh   (needs uv)
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf site build_wasm
mkdir -p build_wasm

uv build --wheel -o build_wasm
wheel=$(basename build_wasm/pyihw-*.whl)

uvx --with mkdocstrings-python zensical build --clean

# The browser installs pyihw from the wheel served next to the app, so drop the
# local-path dependency (and uv source) from the notebook's inline metadata.
grep -v -e '^#     "pyihw",$' -e '^# \[tool.uv.sources\]$' -e '^# pyihw = ' \
    examples/ihw_toy_example.py > build_wasm/ihw_toy_example.py

uvx marimo export html-wasm --mode run --no-show-code \
    build_wasm/ihw_toy_example.py -o site/app

cp "build_wasm/$wheel" site/app/
echo "$wheel" > site/app/wheel.txt
rm -f site/app/CLAUDE.md
touch site/.nojekyll
rm -rf build_wasm
