## Setup

```bash
# 1) Point CUTLASS_DIR to your local CUTLASS checkout (replace with your path)
export CUTLASS_DIR=~/cutlass

# 2) Install Python dependencies (run from the repo root)
pip install -r requirements.txt

# 3) Build/install fouroversix
cd fouroversix
export CUDA_ARCHS=120
export FORCE_BUILD=1
pip install --no-build-isolation -e .
```
