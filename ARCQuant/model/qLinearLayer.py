import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from quantize import *

_KERNEL_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels", "build")
if _KERNEL_BUILD_DIR not in sys.path:
    sys.path.append(_KERNEL_BUILD_DIR)
try:
    import agemm 
except ImportError:
    agemm = None

import math
import random


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def NVFP4_reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w).float() / (448.0*6.0)
    qw, scale_w = agemm.reorder_quantize_w(w/scale, reorder_index, select_num)
    return qw, scale_w, scale
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        select_num, 
        reorder_index,
        out_reorder_index=None,
        quant_type='NVFP4',
        kernel_mode: str = "real",
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.select_num = select_num
        self.quant_type = quant_type
        self.kernel_mode = kernel_mode.strip().lower()
        if self.kernel_mode not in {"real", "pseudo"}:
            raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'real' or 'pseudo'.")

        if self.quant_type == 'NVFP4':
            w = originalLayer.weight.data
            if self.kernel_mode == "real":
                self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w(w, reorder_index.to(torch.int16).cuda(), select_num)
                self.W_fp = None
            else:
                self.W, self.scale_w, self.scale = fake_reorder_quantize_w(w, reorder_index, select_num, dtype='NVFP4')
                self.W_fp = self.W
        else:
            self.W, self.scale_w, self.scale = fake_reorder_quantize_w(originalLayer.weight.data, torch.arange(self.in_features), 0, dtype=quant_type)
            self.W_fp = self.W

        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        qx, scale_x, scale, bsz, q_len = x

        if self.quant_type == 'NVFP4' and self.kernel_mode == "real":
            y = agemm.matmul(qx, self.W, scale_x, self.scale_w, scale * self.scale)
        else:
            w_fp = self.W_fp if self.W_fp is not None else self.W
            y = F.linear(qx, w_fp) * scale * self.scale
        
        torch.cuda.synchronize()
        if self.bias is not None:
            y = y + self.bias

        y = y.reshape(bsz, q_len, -1)
        return y

    