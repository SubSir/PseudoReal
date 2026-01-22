import lm_eval
from lm_eval.models.huggingface import HFLM

import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import torch.nn as nn
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

def load_model(
    model_path: str, device_map: str = None, dtype=torch.float32, **kwargs
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype, **kwargs
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model path",
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("Loading model and tokenizer...")
    model = load_model(args.model, device_map=device, dtype=dtype)
    
    model.to(device)
    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["arc_easy", "arc_challenge", "hellaswag", "boolq"],
        num_fewshot=0,
        device="cuda:0",
    )

    print(results["results"])


if __name__ == "__main__":
    main()