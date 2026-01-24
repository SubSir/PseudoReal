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

cd ARCQuant/kernels
bash remake.sh

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

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

python non_reasoning.py \
  --backend fp_quant \
  --model ./export_fpquant/llama3-8b-nvfp4-gptq \
  --kernel-1 real \
  --kernel-2 pseudo

python ARCQuant/reorder_indices.py --model SubSir/Meta-Llama-3-8B --dataset wikitext2 --act_sort_metric max --samples 128 --seqlen 2048

mv ./saved ./ARCQuant/saved
python non_reasoning.py \
  --backend arcquant \
  --model SubSir/Meta-Llama-3-8B \
  --kernel-1 real \
  --kernel-2 pseudo
```
