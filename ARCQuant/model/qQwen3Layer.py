import torch
from torch import nn
from typing import Optional, Tuple

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from qLinearLayer import QLinearLayer
from qQwenLayer import repeat_kv, reorder_quantize_x


def apply_rotary_pos_emb_qwen3(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (torch.cat((-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), dim=-1) * sin)
    k_embed = (k * cos) + (torch.cat((-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), dim=-1) * sin)
    return q_embed, k_embed


@torch.no_grad()
def quantize_int_group(w, nbits, group_size):
    saved_shape = w.shape
    w = w.reshape(-1, group_size)
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    q_max = (2**nbits) - 1
    q_min = 0
    scales = (w_max - w_min).clamp(min=1e-5) / q_max
    base = torch.round(-w_min / scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    return w.reshape(saved_shape)


class QQwen3RMSNorm(nn.Module):
    def __init__(self, original_norm: Qwen3RMSNorm):
        super().__init__()
        self.original_norm = original_norm

    @torch.no_grad()
    def forward(self, hidden_states):
        return self.original_norm(hidden_states)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.original_norm = self.original_norm.to(*args, **kwargs)
        return self


class QQwen3MLP(nn.Module):
    def __init__(
        self,
        original_mlp: Qwen3MLP,
        select_nums,
        reorder_index,
        layer_idx,
        quant_type,
        kernel_mode: str = "real",
    ):
        super().__init__()
        self.quant_type = quant_type
        self.kernel_mode = kernel_mode.strip().lower()
        if self.kernel_mode not in {"real", "pseudo"}:
            raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'real' or 'pseudo'.")

        name_template = "layers.{}.{}.{}.{}"
        self.gate_proj = QLinearLayer(
            original_mlp.gate_proj,
            select_num=select_nums[name_template.format(layer_idx, "mlp", "gate_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "mlp", "gate_proj", "input")],
            out_reorder_index=reorder_index[name_template.format(layer_idx, "mlp", "down_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.down_proj = QLinearLayer(
            original_mlp.down_proj,
            select_num=select_nums[name_template.format(layer_idx, "mlp", "down_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "mlp", "down_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.up_proj = QLinearLayer(
            original_mlp.up_proj,
            select_num=select_nums[name_template.format(layer_idx, "mlp", "up_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "mlp", "up_proj", "input")],
            out_reorder_index=reorder_index[name_template.format(layer_idx, "mlp", "down_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.act_fn = original_mlp.act_fn

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        bsz, q_len, _ = x.shape
        x = x.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            x, self.up_reorder_index, self.up_proj.select_num, self.quant_type, self.kernel_mode
        )
        torch.cuda.synchronize()
        x = (qx, scale_x, scale, bsz, q_len)
        tmp_result = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        bsz, q_len, _ = tmp_result.shape
        tmp_result = tmp_result.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            tmp_result, self.down_reorder_index, self.down_proj.select_num, self.quant_type, self.kernel_mode
        )
        tmp_result = (qx, scale_x, scale, bsz, q_len)
        return self.down_proj(tmp_result)


class QQwen3Attention(nn.Module):
    def __init__(
        self,
        original_attn: Qwen3Attention,
        kv_cache,
        select_nums,
        reorder_index,
        layer_idx,
        quant_type,
        kernel_mode: str = "real",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.quant_type = quant_type
        self.kernel_mode = kernel_mode.strip().lower()
        if self.kernel_mode not in {"real", "pseudo"}:
            raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'real' or 'pseudo'.")

        self.q_kv_cache = kv_cache
        self.config = original_attn.config
        self.hidden_size = getattr(self.config, "hidden_size", original_attn.o_proj.out_features)
        self.num_heads = getattr(self.config, "num_attention_heads", original_attn.q_proj.out_features // original_attn.head_dim)
        self.head_dim = original_attn.head_dim
        self.num_key_value_heads = getattr(self.config, "num_key_value_heads", original_attn.k_proj.out_features // self.head_dim)
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.attention_dropout = original_attn.attention_dropout
        self.sliding_window = getattr(original_attn, "sliding_window", None)
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        name_template = "layers.{}.{}.{}.{}"
        self.q_proj = QLinearLayer(
            original_attn.q_proj,
            select_num=select_nums[name_template.format(layer_idx, "self_attn", "q_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "self_attn", "q_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.k_proj = QLinearLayer(
            original_attn.k_proj,
            select_num=select_nums[name_template.format(layer_idx, "self_attn", "k_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "self_attn", "k_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.v_proj = QLinearLayer(
            original_attn.v_proj,
            select_num=select_nums[name_template.format(layer_idx, "self_attn", "v_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "self_attn", "v_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.o_proj = QLinearLayer(
            original_attn.o_proj,
            select_num=select_nums[name_template.format(layer_idx, "self_attn", "o_proj", "input")],
            reorder_index=reorder_index[name_template.format(layer_idx, "self_attn", "o_proj", "input")],
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.q_norm = self.q_norm.to(*args, **kwargs)
        self.k_norm = self.k_norm.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        hidden_states = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            hidden_states, self.q_reorder_index, self.q_proj.select_num, self.quant_type, self.kernel_mode
        )
        torch.cuda.synchronize()

        hidden_states = (qx, scale_x, scale, bsz, q_len)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen3(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=128)

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape(bsz * q_len, -1).contiguous().detach()
        qx, scale_x, scale = reorder_quantize_x(
            attn_output, self.o_reorder_index, self.o_proj.select_num, self.quant_type, self.kernel_mode
        )
        torch.cuda.synchronize()
        attn_output = self.o_proj((qx, scale_x, scale, bsz, q_len))
        attn_weights = None
        return attn_output, attn_weights


class QQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        original_layer: Qwen3DecoderLayer,
        kv_cache,
        select_nums,
        reorder_index,
        layer_idx,
        quant_type,
        kernel_mode: str = "real",
    ):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.kernel_mode = kernel_mode.strip().lower()
        if self.kernel_mode not in {"real", "pseudo"}:
            raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'real' or 'pseudo'.")

        self.self_attn = QQwen3Attention(
            original_layer.self_attn,
            kv_cache,
            select_nums=select_nums,
            reorder_index=reorder_index,
            layer_idx=layer_idx,
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.mlp = QQwen3MLP(
            original_layer.mlp,
            select_nums=select_nums,
            reorder_index=reorder_index,
            layer_idx=layer_idx,
            quant_type=quant_type,
            kernel_mode=self.kernel_mode,
        )
        self.input_layernorm = QQwen3RMSNorm(original_layer.input_layernorm)
        self.post_attention_layernorm = QQwen3RMSNorm(original_layer.post_attention_layernorm)
        self.attention_type = original_layer.attention_type

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
