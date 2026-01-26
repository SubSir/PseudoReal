## Setup

### Install & Build

```bash
# 1) Point CUTLASS_DIR to your local CUTLASS checkout (replace with your path)
export CUTLASS_DIR=~/cutlass

# 2) Install Python dependencies (run from the repo root)
pip install -r requirements.txt

# 3) Build/install fouroversix (run from the repo root)
cd fouroversix
export CUDA_ARCHS=120
export FORCE_BUILD=1
pip install --no-build-isolation -e .
cd ..

# 4) Build ARCQuant kernels (run from the repo root)
cd ARCQuant/kernels
bash remake.sh
cd ../..

# 5) Install FP-Quant inference_lib (run from the repo root)
cd FP-Quant/inference_lib
pip install -e .
cd ../..

# 6) Recommended runtime env vars
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

### FP-Quant: Export a Quantized Model (Example)

```bash
python FP-Quant/model_quant.py \
  --model_name_or_path SubSir/Meta-Llama-3-8B \
  --dataset_name_or_path fineweb-edu \
  --sequence_length 2048 \
  --num_sequences 128 \
  --gptq \
  --format nvfp \
  --w_bits 4 --a_bits 4 \
  --w_group_size 16 --a_group_size 16 \
  --transform_class hadamard \
  --hadamard_group_size 16 \
  --export_quantized_model realquant \
  --save_path ./export_fpquant/llama3-8b-nvfp4-gptq \
  --cpu_offload_activations \
  --cpu_offload_modules \
  --fuse_global_scale \
  --amp
```

### Evaluation / Comparison (Example)

```bash
# FP-Quant backend: real vs pseudo
python non_reasoning.py \
  --backend fp_quant \
  --model SubSir/Meta-Llama-3-8B \
  --kernel-1 real \
  --kernel-2 pseudo

# ARCQuant backend: generate reorder_indices first, then move the output "saved" dir to ARCQuant/saved
python ARCQuant/reorder_indices.py \
  --model SubSir/Meta-Llama-3-8B \
  --dataset wikitext2 \
  --act_sort_metric max \
  --samples 128 \
  --seqlen 2048

mv ./saved ./ARCQuant/saved

python non_reasoning.py \
  --backend arcquant \
  --model SubSir/Meta-Llama-3-8B \
  --kernel-1 real \
  --kernel-2 pseudo
```

## Results

Batch size: `16`

### fp_quant

| kernel | arc_challenge (4/6) | arc_easy | boolq | hellaswag |
| --- | ---: | ---: | ---: | ---: |
| pseudo | 48.72 | 77.78 | 77.22 | 58.29 |
| real | 48.38 | 77.65 | 77.34 | 58.38 |

### arc_quant

| kernel | arc_challenge | arc_easy | boolq | hellaswag |
| --- | ---: | ---: | ---: | ---: |
| pseudo | 47.61 | 78.16 | 77.98 | 58.46 |
| real | 47.18 | 78.33 | 80.00 | 58.166 |

### mr-gptq

| kernel | arc_challenge | arc_easy | boolq | hellaswag |
| --- | ---: | ---: | ---: | ---: |
| pseudo | 46.50 | 78.16 | 76.73 | 58.23 |
| real | 44.37 | 77.02 | 73.55 | 57.76 |

## Tests

(If the repo includes unit tests/benchmarks, put the commands here. The original README content did not include concrete test commands.)
