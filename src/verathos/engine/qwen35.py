"""A single-pass Qwen3.5 text backbone for decision models: token ids in, last hidden states out.

Decision models read every token of a request once and generate nothing, so the backbone needs no cache, no decoding
step and no masks: a batch is right-padded, and because both mixers are causal (the gated DeltaNet recurrence and the
causal convolution in the linear layers, causal attention in the full layers) padding after a sequence never changes
the hidden states of its real tokens. That keeps every pass a fixed-shape computation that CUDA graphs can replay.

Per decoder layer the math is transformers' Qwen3.5 reference, computed with flash-linear-attention's Triton kernels:
- the zero-centred RMSNorms as one fused kernel with weight 1 + w in fp32, each norm also adding the preceding
  block's output to the residual;
- linear attention: the causal convolution with SiLU, the chunked
  gated delta rule with the q/k L2 norm, the decay gate -exp(A_log) * softplus(a + dt_bias) and the beta sigmoid in
  the kernel and the key heads shared by their value heads, and the gated RMSNorm;
- full attention: fused q/k RMSNorms, rotary embeddings on the first
  ``rotary_dim`` channels, scaled-dot-product attention with grouped key/value heads, and the sigmoid output gate;
- MLP: the fused SwiGLU.
The fused kernels keep fp32 where the reference rounds to bf16 between steps, so they agree with it to bf16 rounding.

Text requests use one position per token (all three rotary axes equal). Requests with images or videos pass
``inputs_embeds`` with the vision features merged in and 3-axis ``position_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fla.modules.activations import swiglu
from fla.modules.conv import causal_conv1d
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.modules.layernorm import rms_norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

FLA_VERSION = "0.5.2"  # the flash-linear-attention release these kernels were validated against


class _OneTokenClass:
    """A Triton kernel launched with NB=1. fla passes these kernels NB, a class of the token count (rows / 1024 or
    2048, rounded up), as a constexpr and an autotuning key, but their bodies never read it: every pass with a new
    class compiled and autotuned them again, stalling serving for seconds. With NB fixed they tune once per width."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *args, **kwargs: launch(*args, **{**kwargs, "NB": 1})


def _fix_token_classes() -> None:
    import fla.modules.conv.triton.ops as conv
    import fla.modules.fused_norm_gate as gated
    import fla.modules.l2norm as l2
    import fla.modules.layernorm as norm

    for module, name in ((conv, "causal_conv1d_fwd_kernel"), (gated, "layer_norm_gated_fwd_kernel"),
                         (l2, "l2norm_fwd_kernel"), (norm, "layer_norm_fwd_kernel")):
        if not isinstance(getattr(module, name), _OneTokenClass):
            setattr(module, name, _OneTokenClass(getattr(module, name)))


@dataclass
class LinearLayer:
    qkv: torch.Tensor
    z: torch.Tensor
    b: torch.Tensor
    a: torch.Tensor
    conv_weight: torch.Tensor  # [conv_dim, kernel]
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    out_proj: torch.Tensor


@dataclass
class AttentionLayer:
    q: torch.Tensor  # queries and their output gate
    k: torch.Tensor
    v: torch.Tensor
    q_norm: torch.Tensor  # 1 + w, fp32
    k_norm: torch.Tensor
    o_proj: torch.Tensor


@dataclass
class DecoderLayer:
    mixer: LinearLayer | AttentionLayer
    input_norm: torch.Tensor  # 1 + w, fp32
    post_norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


class Backbone:
    """The weights of a ``Qwen3_5TextModel`` arranged for the fused kernels; ``forward`` is the single pass.

    The backbone references the module's weights without copying them (only the norm weights become fp32 1 + w),
    so a 9B model fits a 24 GB GPU with room for activations. The transformers layers are dropped."""

    def __init__(self, text_model) -> None:
        import fla

        if fla.__version__ != FLA_VERSION:
            raise RuntimeError(
                f"the Qwen3.5 kernels need flash-linear-attention=={FLA_VERSION}, found {fla.__version__}"
            )
        _fix_token_classes()
        config = text_model.config
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.k_head_dim = config.linear_key_head_dim
        self.v_head_dim = config.linear_value_head_dim
        self.key_dim = config.linear_key_head_dim * config.linear_num_key_heads
        self.value_dim = config.linear_value_head_dim * config.linear_num_value_heads
        rope = config.rope_parameters
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        self.inv_freq = 1.0 / (rope["rope_theta"] ** exponents)
        self.mrope_section = rope.get("mrope_section")
        self.embed = text_model.embed_tokens.weight
        self.final_norm = 1.0 + text_model.norm.weight.float()
        self.layers = [self._layer(layer) for layer in text_model.layers]
        del text_model.layers
        self.inv_freq = self.inv_freq.to(self.embed.device)
        self.dtype = self.embed.dtype

    def _layer(self, layer) -> DecoderLayer:
        if layer.block_type == "linear_attention":
            m = layer.linear_attn
            mixer = LinearLayer(
                qkv=m.in_proj_qkv.weight,
                z=m.in_proj_z.weight,
                b=m.in_proj_b.weight,
                a=m.in_proj_a.weight,
                conv_weight=m.conv1d.weight.squeeze(1).contiguous(),
                a_log=m.A_log.float(),
                dt_bias=m.dt_bias.float(),
                norm_weight=m.norm.weight,
                out_proj=m.out_proj.weight,
            )
        else:
            m = layer.self_attn
            mixer = AttentionLayer(
                q=m.q_proj.weight,
                k=m.k_proj.weight,
                v=m.v_proj.weight,
                q_norm=1.0 + m.q_norm.weight.float(),
                k_norm=1.0 + m.k_norm.weight.float(),
                o_proj=m.o_proj.weight,
            )
        return DecoderLayer(
            mixer=mixer,
            input_norm=1.0 + layer.input_layernorm.weight.float(),
            post_norm=1.0 + layer.post_attention_layernorm.weight.float(),
            gate=layer.mlp.gate_proj.weight,
            up=layer.mlp.up_proj.weight,
            down=layer.mlp.down_proj.weight,
        )

    def rotary(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin [B, T, rotary_dim] for positions [B, T] (text) or [3, B, T] (temporal, height, width)."""
        if position_ids.ndim == 2:
            freqs = position_ids[..., None].float() * self.inv_freq
        else:
            axes = position_ids[..., None].float() * self.inv_freq  # [3, B, T, rotary_dim / 2]
            freqs = axes[0].clone()
            for axis in (1, 2):  # interleaved: channel i carries axis i % 3 within each section
                channels = slice(axis, self.mrope_section[axis] * 3, 3)
                freqs[..., channels] = axes[axis][..., channels]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [B, heads, T, head_dim]; rotate the first rotary_dim channels."""
        rot, rest = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        cos, sin = cos[:, None], sin[:, None]
        half = self.rotary_dim // 2
        rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
        return torch.cat((rot * cos + rotated * sin, rest), dim=-1)

    def _linear(self, m: LinearLayer, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        mixed, z, b, a = F.linear(h, m.qkv), F.linear(h, m.z), F.linear(h, m.b), F.linear(h, m.a)
        mixed, _ = causal_conv1d(mixed, m.conv_weight, None, activation="silu")
        q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), -1)
        out, _ = chunk_gated_delta_rule(
            q.reshape(B, T, -1, self.k_head_dim),
            k.reshape(B, T, -1, self.k_head_dim),
            v.reshape(B, T, -1, self.v_head_dim),
            g=a,
            beta=b,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=m.a_log,
            dt_bias=m.dt_bias,
            use_beta_sigmoid_in_kernel=True,
        )
        z = z.reshape(B, T, -1, self.v_head_dim)
        out = rms_norm_gated(out, z, m.norm_weight, None, activation="swish", eps=self.eps)
        return F.linear(out.reshape(B, T, -1), m.out_proj)

    def _attention(self, m: AttentionLayer, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        q_gate, k, v = F.linear(h, m.q), F.linear(h, m.k), F.linear(h, m.v)
        q, gate = q_gate.reshape(B, T, -1, 2 * self.head_dim).chunk(2, -1)
        q = rms_norm(q, m.q_norm, None, eps=self.eps).transpose(1, 2)
        k = rms_norm(k.reshape(B, T, -1, self.head_dim), m.k_norm, None, eps=self.eps).transpose(1, 2)
        v = v.reshape(B, T, -1, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.head_dim**-0.5, enable_gqa=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return F.linear(out * torch.sigmoid(gate.reshape(B, T, -1)), m.o_proj)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Last hidden states [B, T, hidden] (after the final norm) for right-padded token ids [B, T]."""
        x = F.embedding(input_ids, self.embed) if inputs_embeds is None else inputs_embeds
        B, T, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).expand(B, T)
        cos, sin = self.rotary(position_ids)
        mlp = None  # the previous layer's MLP output, added to the residual inside the next norm
        for layer in self.layers:
            if mlp is None:
                h = rms_norm(x, layer.input_norm, None, eps=self.eps)
            else:
                h, x = rms_norm(mlp, layer.input_norm, None, residual=x, eps=self.eps, prenorm=True)
            if isinstance(layer.mixer, LinearLayer):
                h = self._linear(layer.mixer, h)
            else:
                h = self._attention(layer.mixer, h, cos, sin)
            h, x = rms_norm(h, layer.post_norm, None, residual=x, eps=self.eps, prenorm=True)
            mlp = F.linear(swiglu(F.linear(h, layer.gate), F.linear(h, layer.up)), layer.down)
        return rms_norm(mlp, self.final_norm, None, residual=x, eps=self.eps)
