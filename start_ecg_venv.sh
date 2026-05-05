#!/usr/bin/env bash
set -e

cd /workspace/ECG

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  pip install -r requirements.txt
  pip install ipykernel jupyterlab
  python -m ipykernel install --user --name ecg_venv --display-name "ECG (.venv)"
else
  source .venv/bin/activate
fi

exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root