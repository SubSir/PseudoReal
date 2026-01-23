import lm_eval
from lm_eval.models.huggingface import HFLM

import torch
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

from fouroversix import apply_ptq, QuantizeBackend

def load_model(
    model_path: str, device_map: str | None = None, dtype=torch.float32, **kwargs
) -> nn.Module:
    if device_map is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, torch_dtype=dtype, **kwargs
        )
    return model


def _evaluate(model: nn.Module, tokenizer, *, batch_size: int) -> dict:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["arc_easy", "arc_challenge", "hellaswag", "boolq"],
        num_fewshot=0,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return results["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model path",
        default="SubSir/Meta-Llama-3-8B",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument(
        "--ptq-backend-1",
        default="real",
        type=str,
        help=(
            "Backend for first apply_ptq run. One of: auto/none, real, pseudo, "
            "pytorch, triton, transformer_engine, cuda."
        ),
    )
    parser.add_argument(
        "--ptq-backend-2",
        default="pseudo",
        type=str,
        help=(
            "Backend for second apply_ptq run. One of: auto/none, real, pseudo, "
            "pytorch, triton, transformer_engine, cuda."
        ),
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def _parse_backend(val: str | None) -> QuantizeBackend | None:
        if val is None:
            return None
        v = val.strip().lower()
        if v in {"auto", "none"}:
            return None
        if v in {"real", "kernel", "kernels"}:
            return QuantizeBackend.triton
        if v in {"pseudo", "ref", "reference"}:
            return QuantizeBackend.pytorch
        return QuantizeBackend(v)

    backend1 = _parse_backend(args.ptq_backend_1)
    backend2 = _parse_backend(args.ptq_backend_2)

    print("Run 1: loading model...")
    model1 = load_model(args.model, device_map=device, dtype=dtype)
    model1.to(device)
    model1.eval()

    if backend1 is not None:
        print(f"Run 1: apply_ptq (backend={backend1}) ...")
    else:
        print("Run 1: apply_ptq (backend=auto) ...")
    apply_ptq(model1, quantize_backend=backend1)

    print("Run 1: evaluating...")
    results1 = _evaluate(model1, tokenizer, batch_size=args.batch_size)
    print({"run": 1, "ptq_backend": args.ptq_backend_1, "results": results1})

    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Run 2: loading model...")
    model2 = load_model(args.model, device_map=device, dtype=dtype)
    model2.to(device)
    model2.eval()

    if backend2 is not None:
        print(f"Run 2: apply_ptq (backend={backend2}) ...")
    else:
        print("Run 2: apply_ptq (backend=auto) ...")
    apply_ptq(model2, quantize_backend=backend2)

    print("Run 2: evaluating...")
    results2 = _evaluate(model2, tokenizer, batch_size=args.batch_size)
    print({"run": 2, "ptq_backend": args.ptq_backend_2, "results": results2})


if __name__ == "__main__":
    main()