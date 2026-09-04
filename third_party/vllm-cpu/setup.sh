#!/bin/sh
# Build vLLM's CPU backend from source into its own venv (no root, AVX2 ok).
set -eu
cd "$(dirname "$0")"
export UV_LINK_MODE=copy
[ -d .venv ] || uv venv .venv --python 3.12 --seed
. .venv/bin/activate
if [ ! -d vllm ]; then
  git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm.git
fi
cd vllm
uv pip install --upgrade "cmake>=3.26" wheel packaging ninja "setuptools>=77" "setuptools-scm>=8" numpy
uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=cpu MAX_JOBS=32 uv pip install -e . --no-build-isolation -v
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
