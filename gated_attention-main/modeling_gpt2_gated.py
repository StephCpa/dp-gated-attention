from typing import Optional

import torch
import torch.nn as nn
from transformers.models.gpt2 import modeling_gpt2 as gpt2
from transformers.pytorch_utils import Conv1D
try:
    from transformers.utils import deprecate_kwarg
except ImportError:
    def deprecate_kwarg(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator


ENCODER_DECODER_CACHE = getattr(gpt2, "EncoderDecoderCache", None)

try:
    from opacus.grad_sample import register_grad_sampler

    @register_grad_sampler(Conv1D)
    def _conv1d_grad_sampler(layer, activations, backprops):
        activations = activations[0]
        ret = {}
        if layer.weight.requires_grad:
            ret[layer.weight] = torch.einsum("n...i,n...j->nij", activations, backprops)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...j->nj", backprops)
        return ret
except Exception:
    pass


class GatedGPT2Attention(gpt2.GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, gate_type="headwise"):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        if gate_type not in {"headwise", "elementwise", "none"}:
            raise ValueError("gate_type must be one of: headwise, elementwise, none")
        self.gate_type = gate_type
        self._gate_dim = 0
        if gate_type == "headwise":
            self._gate_dim = self.num_heads
        elif gate_type == "elementwise":
            self._gate_dim = self.embed_dim
        self._fused_gate = self._gate_dim > 0 and not is_cross_attention
        if self._fused_gate:
            self.c_attn = Conv1D(3 * self.embed_dim + self._gate_dim, self.embed_dim)

    def init_fused_c_attn(self, base_attn):
        if not self._fused_gate:
            return
        if not hasattr(base_attn, "c_attn"):
            raise ValueError("base_attn must have c_attn to initialize fused gating.")
        if base_attn.c_attn.weight.shape[1] != 3 * self.embed_dim:
            raise ValueError("base_attn.c_attn has unexpected shape; cannot init fused gate.")
        q_end = self.embed_dim
        gate_end = self.embed_dim + self._gate_dim
        k_end = gate_end + self.embed_dim
        v_end = k_end + self.embed_dim
        with torch.no_grad():
            self.c_attn.weight.zero_()
            self.c_attn.bias.zero_()
            self.c_attn.weight[:, :q_end] = base_attn.c_attn.weight[:, :q_end]
            self.c_attn.bias[:q_end] = base_attn.c_attn.bias[:q_end]
            self.c_attn.weight[:, gate_end:k_end] = base_attn.c_attn.weight[:, q_end : q_end + self.embed_dim]
            self.c_attn.bias[gate_end:k_end] = base_attn.c_attn.bias[q_end : q_end + self.embed_dim]
            self.c_attn.weight[:, k_end:v_end] = base_attn.c_attn.weight[:, q_end + self.embed_dim : q_end + 2 * self.embed_dim]
            self.c_attn.bias[k_end:v_end] = base_attn.c_attn.bias[q_end + self.embed_dim : q_end + 2 * self.embed_dim]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if past_key_values is None and "layer_past" in kwargs:
            past_key_values = kwargs.pop("layer_past")

        if self.gate_type == "none":
            return super().forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            return super().forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        if past_key_values is not None:
            if ENCODER_DECODER_CACHE is not None and isinstance(past_key_values, ENCODER_DECODER_CACHE):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if not self._fused_gate:
            raise ValueError("Fused gating requested but c_attn was not expanded.")

        fused = self.c_attn(hidden_states)
        query_states, gate_score, key_states, value_states = fused.split(
            [self.embed_dim, self._gate_dim, self.embed_dim, self.embed_dim], dim=2
        )
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            if hasattr(curr_past_key_value, "update"):
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention and ENCODER_DECODER_CACHE is not None:
                    past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface = gpt2.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = gpt2.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        bsz, q_len, _, _ = attn_output.size()
        if self.gate_type == "headwise":
            gate_score = gate_score.view(bsz, q_len, self.num_heads, 1)
        else:
            gate_score = gate_score.view(bsz, q_len, self.num_heads, self.head_dim)
        if attn_output.shape[1] == self.num_heads:
            gate_score = gate_score.permute(0, 2, 1, 3)
        attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


def apply_gpt2_gated_attention(model, gate_type="headwise"):
    if gate_type not in {"headwise", "elementwise", "none"}:
        raise ValueError("gate_type must be one of: headwise, elementwise, none")

    model.config.headwise_attn_output_gate = gate_type == "headwise"
    model.config.elementwise_attn_output_gate = gate_type == "elementwise"

    if gate_type == "none":
        return model

    for block in model.transformer.h:
        old_attn = block.attn
        gated = GatedGPT2Attention(
            config=model.config,
            is_cross_attention=False,
            layer_idx=old_attn.layer_idx,
            gate_type=gate_type,
        )
        gated.load_state_dict(old_attn.state_dict(), strict=False)
        gated.init_fused_c_attn(old_attn)
        block.attn = gated

    return model
