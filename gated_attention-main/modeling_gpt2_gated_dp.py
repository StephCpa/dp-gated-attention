"""
modeling_gpt2_gated_dp.py
=========================
DP-aware gated attention for GPT-2.

Key differences from the original modeling_gpt2_gated.py:

1. Noise-aware gate initialisation
   The original init_fused_c_attn() zeros the gate bias, which gives
   sigmoid(0) = 0.5 at step 0 — maximum uncertainty, maximum noise
   pass-through.  Under DP-SGD the noise level is highest at the very
   start of training (privacy budget drains fastest early on), so we
   want the gate to be maximally suppressive at initialisation and
   gradually open up.

   We set the gate bias to `init_gate_bias` (default -2.197) so that
       sigmoid(-2.197) ≈ 0.1
   i.e. the gate suppresses ~90% of the attention output on step 0.
   The model then learns to selectively open heads as training proceeds.

2. Gate L1 regularisation helper
   gate_l1_loss() returns the mean L1 norm of the gate weight slice and
   gate bias so the caller can add   λ * gate_l1_loss()   to the task
   loss.  The caller is responsible for scheduling λ.

3. Gate sparsity measurement
   gate_sparsity(x, threshold=0.1) does a forward pass and returns the
   fraction of gate activations below `threshold`, which serves as the
   empirical noise-suppression rate from RQ1.
"""

import math
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

ENCODER_DECODER_CACHE = getattr(gpt2, "EncoderDecoderCache", None)

# sigmoid^{-1}(0.1) = ln(0.1/0.9) ≈ -2.197
# Use this as the default so the gate suppresses ~90% of outputs at init.
_DP_INIT_BIAS: float = math.log(0.1 / 0.9)  # ≈ -2.197


class DPGatedGPT2Attention(gpt2.GPT2Attention):
    """
    GPT-2 self-attention with a query-dependent sigmoid gate after SDPA,
    initialised for DP training.

    Parameters
    ----------
    config : GPT-2 config
    gate_type : "headwise" | "elementwise" | "none"
        headwise  — one gate scalar per head per token   (gate_dim = num_heads)
        elementwise — one gate value per feature per token (gate_dim = embed_dim)
        none      — no gate; behaves identically to the base GPT2Attention
    init_gate_bias : float
        Initial value for the gate projection bias.
        Default _DP_INIT_BIAS ≈ -2.197 → sigmoid ≈ 0.1 at step 0.
        Use 0.0 to reproduce the original (naive) behaviour.
    """

    def __init__(
        self,
        config,
        is_cross_attention: bool = False,
        layer_idx: Optional[int] = None,
        gate_type: str = "headwise",
        init_gate_bias: float = _DP_INIT_BIAS,
    ):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

        if gate_type not in {"headwise", "elementwise", "none"}:
            raise ValueError("gate_type must be one of: headwise, elementwise, none")

        self.gate_type = gate_type
        self.init_gate_bias = init_gate_bias
        self._gate_dim = 0

        if gate_type == "headwise":
            self._gate_dim = self.num_heads
        elif gate_type == "elementwise":
            self._gate_dim = self.embed_dim

        self._fused_gate = self._gate_dim > 0 and not is_cross_attention
        if self._fused_gate:
            self.c_attn = Conv1D(3 * self.embed_dim + self._gate_dim, self.embed_dim)

    # ------------------------------------------------------------------
    # Helpers to locate the gate slice inside the fused c_attn matrix
    # ------------------------------------------------------------------

    def _gate_slice(self):
        """Return (q_end, gate_end) indices for the gate in c_attn.bias."""
        q_end = self.embed_dim
        gate_end = self.embed_dim + self._gate_dim
        return q_end, gate_end

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_fused_c_attn(self, base_attn):
        """
        Initialise the fused c_attn weight by copying Q/K/V from a
        pretrained base attention and setting the gate bias to
        self.init_gate_bias (default ≈ -2.197 for DP-aware init).
        """
        if not self._fused_gate:
            return
        if not hasattr(base_attn, "c_attn"):
            raise ValueError("base_attn must have c_attn attribute.")
        if base_attn.c_attn.weight.shape[1] != 3 * self.embed_dim:
            raise ValueError("base_attn.c_attn has unexpected shape.")

        q_end, gate_end = self._gate_slice()
        k_end = gate_end + self.embed_dim
        v_end = k_end + self.embed_dim

        with torch.no_grad():
            self.c_attn.weight.zero_()
            self.c_attn.bias.zero_()

            # Copy Q
            self.c_attn.weight[:, :q_end] = base_attn.c_attn.weight[:, :q_end]
            self.c_attn.bias[:q_end] = base_attn.c_attn.bias[:q_end]

            # --- Gate bias: DP-aware init ---
            # bias[:q_end] is Q bias (kept from pretrain).
            # bias[q_end:gate_end] is the gate bias — we set this to
            # init_gate_bias so sigmoid(gate) ≈ 0.1 at step 0.
            self.c_attn.bias[q_end:gate_end] = self.init_gate_bias
            # Gate weight stays zero: gate output is purely bias-driven
            # at initialisation.  The model learns non-trivial gating
            # from data as training proceeds.

            # Copy K
            self.c_attn.weight[:, gate_end:k_end] = base_attn.c_attn.weight[
                :, q_end: q_end + self.embed_dim
            ]
            self.c_attn.bias[gate_end:k_end] = base_attn.c_attn.bias[
                q_end: q_end + self.embed_dim
            ]

            # Copy V
            self.c_attn.weight[:, k_end:v_end] = base_attn.c_attn.weight[
                :, q_end + self.embed_dim: q_end + 2 * self.embed_dim
            ]
            self.c_attn.bias[k_end:v_end] = base_attn.c_attn.bias[
                q_end + self.embed_dim: q_end + 2 * self.embed_dim
            ]

    # ------------------------------------------------------------------
    # DP training utilities
    # ------------------------------------------------------------------

    def _get_gate_params(self):
        """Get gate weight and bias, handling both Conv1D and nn.Linear."""
        q_end, gate_end = self._gate_slice()
        gate_b = self.c_attn.bias[q_end:gate_end]
        if isinstance(self.c_attn, Conv1D):
            # Conv1D weight shape: (in_features, out_features)
            gate_w = self.c_attn.weight[:, q_end:gate_end]
        else:
            # nn.Linear weight shape: (out_features, in_features)
            gate_w = self.c_attn.weight[q_end:gate_end, :]
        return gate_w, gate_b

    def gate_l1_loss(self) -> torch.Tensor:
        """
        Mean L1 norm of gate parameters (weight slice + bias slice).

        Add   λ * model_gate_l1_loss(model)   to the task loss to
        encourage gate sparsity.  λ should be scheduled to decrease
        as the privacy budget is consumed (see dp_train_v2.py).
        """
        if not self._fused_gate:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        gate_w, gate_b = self._get_gate_params()
        return gate_w.abs().mean() + gate_b.abs().mean()

    @torch.no_grad()
    def gate_sparsity(self, hidden_states: torch.Tensor, threshold: float = 0.1) -> float:
        """
        Forward a single batch through the gate projection only and
        return the fraction of gate activations below `threshold`.

        This is the empirical proxy for the noise-suppression rate s
        from RQ1.  Higher is better under DP.
        """
        if not self._fused_gate:
            return 0.0
        gate_w, gate_b = self._get_gate_params()
        if isinstance(self.c_attn, Conv1D):
            # Conv1D: weight is (in, out), forward does x @ weight
            gate_score = hidden_states @ gate_w + gate_b
        else:
            # nn.Linear: weight is (out, in), need x @ weight.T
            gate_score = hidden_states @ gate_w.t() + gate_b
        gate_prob = torch.sigmoid(gate_score)
        return (gate_prob < threshold).float().mean().item()

    # ------------------------------------------------------------------
    # Forward pass (identical to original except uses self.init_gate_bias)
    # ------------------------------------------------------------------

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
            if ENCODER_DECODER_CACHE is not None and isinstance(
                past_key_values, ENCODER_DECODER_CACHE
            ):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if not self._fused_gate:
            raise RuntimeError("Fused gate was not set up — call init_fused_c_attn first.")

        fused = self.c_attn(hidden_states)
        query_states, gate_score, key_states, value_states = fused.split(
            [self.embed_dim, self._gate_dim, self.embed_dim, self.embed_dim], dim=2
        )
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if past_key_values is not None and not is_cross_attention:
            if hasattr(curr_past_key_value, "update"):
                cache_position_arg = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    {"cache_position": cache_position_arg}
                )
                if is_cross_attention and ENCODER_DECODER_CACHE is not None:
                    past_key_values.is_updated[self.layer_idx] = True

        is_causal = (
            attention_mask is None
            and query_states.shape[-2] > 1
            and not is_cross_attention
        )
        using_eager = self.config._attn_implementation == "eager"
        attention_interface = gpt2.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = gpt2.ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

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


# ------------------------------------------------------------------
# Model-level helpers
# ------------------------------------------------------------------

def apply_dp_gated_attention(
    model,
    gate_type: str = "headwise",
    init_gate_bias: float = _DP_INIT_BIAS,
):
    """
    Replace every GPT2Attention block with DPGatedGPT2Attention in-place.

    Parameters
    ----------
    model : GPT-2 model (from transformers)
    gate_type : "headwise" | "elementwise" | "none"
    init_gate_bias : float
        -2.197 (default, DP-aware) or 0.0 (naive baseline)
    """
    if gate_type not in {"headwise", "elementwise", "none"}:
        raise ValueError("gate_type must be one of: headwise, elementwise, none")

    model.config.headwise_attn_output_gate = gate_type == "headwise"
    model.config.elementwise_attn_output_gate = gate_type == "elementwise"

    if gate_type == "none":
        return model

    for block in model.transformer.h:
        old_attn = block.attn
        gated = DPGatedGPT2Attention(
            config=model.config,
            is_cross_attention=False,
            layer_idx=old_attn.layer_idx,
            gate_type=gate_type,
            init_gate_bias=init_gate_bias,
        )
        # Copy all non-gate state from pretrained weights
        state = {k: v for k, v in old_attn.state_dict().items()
                 if not k.startswith("c_attn.")}
        gated.load_state_dict(state, strict=False)
        gated.init_fused_c_attn(old_attn)
        block.attn = gated

    return model


def convert_conv1d_to_linear(model):
    """
    Convert all HuggingFace Conv1D layers to nn.Linear for Opacus compatibility.

    Conv1D stores weight as (in_features, out_features) and does x @ weight + bias.
    nn.Linear stores weight as (out_features, in_features) and does x @ weight.T + bias.
    The functional result is identical; only the storage layout differs.

    Must be called AFTER apply_dp_gated_attention() (which uses Conv1D internally
    for init_fused_c_attn) but BEFORE Opacus make_private().
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, Conv1D):
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            linear = nn.Linear(in_features, out_features)
            linear.weight.data = module.weight.data.t().contiguous()
            linear.bias.data = module.bias.data.clone()
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], linear)
    return model


def model_gate_l1_loss(model) -> torch.Tensor:
    """Sum of gate_l1_loss() across all DPGatedGPT2Attention layers."""
    total = None
    for module in model.modules():
        if isinstance(module, DPGatedGPT2Attention) and module._fused_gate:
            l = module.gate_l1_loss()
            total = l if total is None else total + l
    if total is None:
        return torch.tensor(0.0)
    return total


@torch.no_grad()
def model_gate_sparsity(model, sample_hidden: torch.Tensor, threshold: float = 0.1) -> float:
    """
    Average gate sparsity across all gated layers given a sample of
    hidden states (shape: B x L x D).

    Call this every N steps with a held-out validation batch to track
    how the gate sparsity evolves during DP training.
    """
    values = []
    for module in model.modules():
        if isinstance(module, DPGatedGPT2Attention) and module._fused_gate:
            values.append(module.gate_sparsity(sample_hidden, threshold=threshold))
    return float(sum(values) / len(values)) if values else 0.0
