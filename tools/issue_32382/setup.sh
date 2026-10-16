#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# Run only on an independently approved host; this script never provisions one.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${1:?Usage: setup.sh /absolute/path/to/new/venv}"
command -v python3.12 >/dev/null
command -v tmux >/dev/null
if [[ ! -d "${venv_dir}" ]]; then
    python3.12 -m venv "${venv_dir}"
fi
"${venv_dir}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
"${venv_dir}/bin/python" -m pip install --requirement "${script_dir}/requirements.txt"
