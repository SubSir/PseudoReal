set -e
python non_reasoning.py   --backend arcquant   --model SubSir/Meta-Llama-3-8B   --kernel-1 pseudo  --kernel-2 real
# python non_reasoning.py   --backend 4o6   --model SubSir/Meta-Llama-3-8B   --kernel-1 real   --kernel-2 pseudo
