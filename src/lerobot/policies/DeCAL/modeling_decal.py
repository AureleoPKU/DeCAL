#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import torch._dynamo as dynamo
from einops import rearrange

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel

from lerobot.policies.DeCAL.cosmos_tokenizer.image_lib import ImageTokenizer
from lerobot.policies.DeCAL.configuration_decal import DeCALConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    OBS_PREFIX, 
    OBS_IMAGES, 
    OBS_TACTILE_IMAGES,
    OPENPI_ATTENTION_MASK_VALUE,
)

from .tactile_utils import (
    MultiFingerTactileEncoder,
    TactileFlowEncoder,
    TactileLocalTokenImageDecoder,
    compute_tactile_flow,
)

DECAL_ROOT = Path(__file__).resolve().parents[4]


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    und_expert,
    gen_expert,
    act_expert,
    tactile_cross_attn_layers,
    local_tactile_tokens=None,
    use_tactile_gate: bool = False,
    tactile_gate=None,
    global_tactile_token=None,
):
    models = [und_expert.language_model, gen_expert, act_expert]
    query_states = []
    key_states = []
    value_states = []
    prefix_ln_for_tactile = None
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        if i == 0 and local_tactile_tokens is not None:
            prefix_ln_for_tactile = hidden_states
        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = und_expert.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_qwen3_vl.eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = out_emb + hidden_states
        if i == 0 and local_tactile_tokens is not None and prefix_ln_for_tactile is not None:
            tac_mha = tactile_cross_attn_layers[layer_idx]
            tactile_ctx, _ = tac_mha(query=prefix_ln_for_tactile, key=local_tactile_tokens, value=local_tactile_tokens)
            if use_tactile_gate and tactile_gate is not None and global_tactile_token is not None:
                gate = tactile_gate(global_tactile_token).view(-1, 1, 1)
                out_emb = out_emb + (gate * tactile_ctx)
            else:
                out_emb = out_emb + tactile_ctx
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = out_emb + after_first_residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:  
    """Configuration for Qwen model variants."""

    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    num_hidden_layers = int(variant.split('_')[-1][:-1])  # pattern: qwen3_vl_28l or qwen3_xxl
    if variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,
            intermediate_size=6144, 
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024,
            intermediate_size=3072, 
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class Qwen3VLWithExpertModel(
    nn.Module
):  
    """Qwen3-VL backbone with generation and action experts for DeCAL."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        use_tactile_gate: bool = False,
    ):
        super().__init__()
        self.use_tactile_gate = use_tactile_gate

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        vlm_config_hf.text_config.tie_word_embeddings = True
        vlm_config_hf.tie_word_embeddings = True
        vlm_config_hf.vision_config.deepstack_visual_indexes=[5, 11, 17]
        vlm_config_hf.vision_config.depth=24
        vlm_config_hf.vision_config.hidden_size=1024
        vlm_config_hf.vision_config.intermediate_size=4096
        vlm_config_hf.vision_config.out_hidden_size=2048
        
        # self.und_expert = Qwen3VLForConditionalGeneration(config=vlm_config_hf)
        self.und_expert = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", 
            config=vlm_config_hf, 
            ignore_mismatched_sizes=True,
            cache_dir=os.environ.get("DECAL_QWEN_CACHE_DIR", str(DECAL_ROOT / "ckpt" / "qwen")),
        )

        gen_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        gen_expert_config_hf.head_dim=action_expert_config.head_dim
        gen_expert_config_hf.hidden_size=action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        gen_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        gen_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.gen_expert = Qwen3VLTextModel(config=gen_expert_config_hf)
        self.gen_expert.embed_tokens = None
        self.gen_expert.lm_head = None

        action_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        action_expert_config_hf.head_dim=action_expert_config.head_dim
        action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        action_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        action_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        action_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.act_expert = Qwen3VLTextModel(config=action_expert_config_hf)
        self.act_expert.embed_tokens = None
        self.act_expert.lm_head = None

        assert self.und_expert.config.text_config.num_hidden_layers == self.act_expert.config.num_hidden_layers
       
        n_layers = self.und_expert.config.text_config.num_hidden_layers
        self.tactile_cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=vlm_config.hidden_size,
                    num_heads=8,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.tactile_gate = (
            nn.Sequential(
                nn.LayerNorm(vlm_config.hidden_size),
                nn.Linear(vlm_config.hidden_size, vlm_config.hidden_size),
                nn.SiLU(),
                nn.Linear(vlm_config.hidden_size, 1),
                nn.Sigmoid(),
            )
            if self.use_tactile_gate
            else None
        )
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            # "visual.patch_embed.proj.weight",
            # "visual.patch_embed.proj.bias",
            # "visual.pos_embed.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _forward_und_language_model_prefix_with_tactile(
        self,
        inputs_embeds: Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        use_cache: bool | None,
        local_tactile_tokens: Tensor,
        global_tactile_token: Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Run und language_model for prefix only, with same tactile fusion as ``compute_layer_complete`` (stream 0)."""
        try:
            from transformers.masking_utils import create_causal_mask
        except ImportError:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        tac_dtype = next(self.tactile_cross_attn_layers[0].parameters()).dtype
        local_tactile_tokens = local_tactile_tokens.to(dtype=tac_dtype)
        if global_tactile_token is not None:
            global_tactile_token = global_tactile_token.to(dtype=tac_dtype)

        lm = self.und_expert.language_model
        config = lm.config

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = lm.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(lm.layers):
            residual = hidden_states
            h = decoder_layer.input_layernorm(hidden_states)
            if decoder_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                h = h.to(dtype=torch.bfloat16)
            attn_out, _ = decoder_layer.self_attn(
                hidden_states=h,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attn_out
            tactile_ctx, _ = self.tactile_cross_attn_layers[layer_idx](
                query=h.to(dtype=tac_dtype),
                key=local_tactile_tokens,
                value=local_tactile_tokens,
            )
            if self.use_tactile_gate and self.tactile_gate is not None and global_tactile_token is not None:
                gate = self.tactile_gate(global_tactile_token).view(-1, 1, 1)
                hidden_states = hidden_states + (gate * tactile_ctx).to(dtype=hidden_states.dtype)
            else:
                hidden_states = hidden_states + tactile_ctx.to(dtype=hidden_states.dtype)
            res_mlp = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            if decoder_layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = res_mlp + hidden_states

        hidden_states = lm.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        local_tactile_tokens: torch.Tensor | None = None,
        global_tactile_token: torch.Tensor | None = None,
    ):
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            if local_tactile_tokens is not None:
                prefix_lm_out = self._forward_und_language_model_prefix_with_tactile(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    local_tactile_tokens=local_tactile_tokens,
                    global_tactile_token=global_tactile_token,
                )
                past_key_values = prefix_lm_out.past_key_values
                prefix_output = prefix_lm_out.last_hidden_state
            else:
                prefix_output = self.und_expert.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
                past_key_values = prefix_output.past_key_values
                prefix_output = prefix_output.last_hidden_state
            middle_output = None
            suffix_output = None
            
        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.gen_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = middle_output.past_key_values
            prefix_output = None
            middle_output = middle_output.last_hidden_state
            suffix_output = None
            
        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.act_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = None
            prefix_output = None
            middle_output = None
            suffix_output = suffix_output.last_hidden_state
        else:
            models = [self.und_expert.language_model, self.gen_expert, self.act_expert]
            num_layers = self.und_expert.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.act_expert, "gradient_checkpointing")
                and self.gen_expert.gradient_checkpointing
                and self.act_expert.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                        tactile_cross_attn_layers=self.tactile_cross_attn_layers,
                        local_tactile_tokens=local_tactile_tokens,
                        use_tactile_gate=self.use_tactile_gate,
                        tactile_gate=self.tactile_gate,
                        global_tactile_token=global_tactile_token,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                        tactile_cross_attn_layers=self.tactile_cross_attn_layers,
                        local_tactile_tokens=local_tactile_tokens,
                        use_tactile_gate=self.use_tactile_gate,
                        tactile_gate=self.tactile_gate,
                        global_tactile_token=global_tactile_token,
                    )

            # final norm
            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            past_key_values = None
            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]

        return [prefix_output, middle_output, suffix_output], past_key_values


class DeCALModel(nn.Module):

    def __init__(self, config: DeCALConfig):
        super().__init__()
        print("use_tactile_deform:", config.use_tactile_deform)
        print("use_tactile_batchnorm:", config.use_tactile_batchnorm)
        self.config = config

        vlm_config = get_qwen_config(config.qwen3_vl_variant)
        action_expert_config = get_qwen_config(config.action_expert_variant)

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            vlm_config,
            action_expert_config,
            precision=config.dtype,
            use_tactile_gate=config.use_tactile_gate,
        )

        cosmos_dir = Path(
            os.environ.get(
                "DECAL_COSMOS_TOKENIZER_DIR",
                str(DECAL_ROOT / "ckpt" / "Cosmos-Tokenizer-CI8x8"),
            )
        )
        if not (cosmos_dir / "encoder.jit").exists():
            logging.warning("Cosmos-Tokenizer-CI8x8 not found, downloading...")
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id="nvidia/Cosmos-Tokenizer-CI8x8", local_dir=cosmos_dir)

        self.cosmos = ImageTokenizer(
            checkpoint_enc=str(cosmos_dir / "encoder.jit"),
            checkpoint_dec=str(cosmos_dir / "decoder.jit"),
        )

        self.tactile_encoder = MultiFingerTactileEncoder(
            in_channels=3,
            num_fingers=10,
            hidden_dim=vlm_config.hidden_size,
            use_batchnorm=config.use_tactile_batchnorm,
        )
        self.tactile_encoder_deform_decoder = TactileLocalTokenImageDecoder(
            token_dim=vlm_config.hidden_size,
            out_channels=3,
            num_fingers=10,
        )

        vae_dim = 16
        gen_proj_dim = action_expert_config.hidden_size
        ds = self.config.scale_factor
        # self.downsample_conv = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # junhao 
        # self.cosmos_in_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vlm_config.hidden_size, kernel_size=1, stride=1, padding=0) # junhao 
        # self.cosmos_out_proj = nn.Conv2d(in_channels=vlm_config.hidden_size, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # junhao 
        # self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # junhao 
        
        self.cosmos_in_proj = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # jia 
        self.downsample_conv = nn.Conv2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # jia
        self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # jia 
        # self.cosmos_out_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=1, stride=1, padding=0) # jia
        self.cosmos_out_proj = nn.Linear(gen_proj_dim, vae_dim)
        self.cosmos_out_layer_norm = nn.LayerNorm(gen_proj_dim) # jia


        # Action embedding / decoding.
        # If split_arm_hand_action=False, fall back to original single-stream behavior.
        self.split_arm_hand_action = config.split_arm_hand_action
        self.action_hidden_dim = action_expert_config.hidden_size
        self.arm_action_dim = config.arm_action_dim

        if self.split_arm_hand_action:
            # Split action space into arm (2x6) and hand (2x22) streams.
            # Each stream has its own in/out projection, while action expert attention still mixes both streams.
            if config.max_action_dim < self.arm_action_dim:
                raise ValueError(
                    f"max_action_dim ({config.max_action_dim}) must be >= arm_action_dim ({self.arm_action_dim})."
                )
            self.hand_action_dim = config.max_action_dim - self.arm_action_dim
            self.arm_action_in_proj = nn.Linear(self.arm_action_dim, self.action_hidden_dim)
            self.hand_action_in_proj = nn.Linear(self.hand_action_dim, self.action_hidden_dim)
            self.arm_action_out_proj = nn.Linear(self.action_hidden_dim, self.arm_action_dim)
            self.hand_action_out_proj = nn.Linear(self.action_hidden_dim, self.hand_action_dim)
        else:
            self.arm_action_dim = config.max_action_dim
            self.hand_action_dim = 0
            # Original single-stream projection
            self.action_in_proj = nn.Linear(config.max_action_dim, self.action_hidden_dim)
            self.action_out_proj = nn.Linear(self.action_hidden_dim, config.max_action_dim)

        # Tactile Gen branch: tactile flow/deform -> tactile tokens (append to middle)
        # -> VLM -> tactile latent (from tactile tokens only)
        if config.use_tactile_gen_branch:
            if config.use_tactile_force:
                self.tactile_flow_encoder = None
                self.tactile_force_encoder = nn.Sequential(
                    nn.Linear(12, action_expert_config.hidden_size),
                    nn.LayerNorm(action_expert_config.hidden_size),
                    nn.SiLU(),
                )
            else:
                tactile_gen_in_channels = 6 if config.use_tactile_deform else 2
                self.tactile_flow_encoder = TactileFlowEncoder(
                    in_channels=tactile_gen_in_channels,
                    num_fingers=10,
                    hidden_dim=action_expert_config.hidden_size,
                    use_batchnorm=config.use_tactile_batchnorm,
                )
                self.tactile_force_encoder = None
            # Per-finger projection: (B, 10, hidden) -> (B, 10, gen_proj_dim)
            self.tactile_token_proj = nn.Linear(action_expert_config.hidden_size, gen_proj_dim)
            self.tactile_latent_proj = nn.Sequential(
                nn.Linear(gen_proj_dim, config.tactile_latent_dim),
                nn.LayerNorm(config.tactile_latent_dim),
            )
            D = config.tactile_latent_dim
            self.tactile_deform_decoder = self._build_tactile_image_decoder(D, 3)
            self.tactile_orig_decoder = self._build_tactile_image_decoder(D, 3)
            # Per-finger latent -> per-finger 6D force: (B, 10, D) -> (B, 10, 6).
            self.force_6d_decoder = nn.Linear(D, 6)
        else:
            self.tactile_flow_encoder = None
            self.tactile_force_encoder = None
            self.tactile_token_proj = None
            self.tactile_latent_proj = None
            self.tactile_deform_decoder = None
            self.tactile_orig_decoder = None
            self.force_6d_decoder = None

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.hidden_size, action_expert_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_config.hidden_size, action_expert_config.hidden_size)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)
        
        self.set_requires_grad()
    
    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwen3_vl_with_expert.und_expert.visual.eval()
            for params in self.qwen3_vl_with_expert.und_expert.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwen3_vl_with_expert.und_expert.eval()
            for params in self.qwen3_vl_with_expert.und_expert.parameters():
                params.requires_grad = False
        
        if self.config.train_vlm_only:
            self.qwen3_vl_with_expert.gen_expert.eval()
            for params in self.qwen3_vl_with_expert.gen_expert.parameters():
                params.requires_grad = False
            self.qwen3_vl_with_expert.act_expert.eval()
            for params in self.qwen3_vl_with_expert.act_expert.parameters():
                params.requires_grad = False
        
        self.cosmos.eval()
        for params in self.cosmos.parameters():
            params.requires_grad = False
    
    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.qwen3_vl_with_expert.und_expert.visual.eval()

        if self.config.train_expert_only:
            self.qwen3_vl_with_expert.und_expert.eval()
        
        self.cosmos.eval()
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = True
        self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = True
        self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = True
        self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for DeCAL model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = False
        self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = False
        self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = False
        self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for DeCAL model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _build_tactile_image_decoder(self, latent_dim: int, out_channels: int):
        """Build decoder: (B, 10, D) -> (B, 10, out_channels, 224, 224)."""
        return TactileLocalTokenImageDecoder(token_dim=latent_dim, out_channels=out_channels, num_fingers=10)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _split_action_parts(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.split_arm_hand_action:
            # Treat all as "arm" stream for backward compatibility.
            return actions, actions.new_zeros(*actions.shape[:-1], 0)
        arm_actions = actions[..., : self.arm_action_dim]
        hand_actions = actions[..., self.arm_action_dim : self.arm_action_dim + self.hand_action_dim]
        return arm_actions, hand_actions

    def _decode_action_parts(self, arm_hidden: torch.Tensor, hand_hidden: torch.Tensor) -> torch.Tensor:
        if not self.split_arm_hand_action:
            return self.action_out_proj(arm_hidden)
        arm_out = self.arm_action_out_proj(arm_hidden)
        hand_out = self.hand_action_out_proj(hand_hidden)
        return torch.cat([arm_out, hand_out], dim=-1)

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values = pixel_values.view(-1, D1)
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs, _ = self.qwen3_vl_with_expert.und_expert.visual(pixel_values, image_grid_thw)

        embs = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens = lang_tokens.view(-1)
        embs[lang_tokens == image_token_id] = image_embs  # replace dummy embeds with image_embeds
        embs = embs.view(B, L, D2)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)

        return embs, pad_masks, att_masks
    
    def get_cosmos_features(self, images):
        shape = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.reshape(-1, c, h, w)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.cosmos.encode(images)
        c, h, w = features.shape[-3:]
        features = features.view(*shape, c, h, w)
        return features
    
    def embed_middle(self, images, img_masks):
        device = images[0].device
        B, N_view, T = images.shape[:3]
        features = self.get_cosmos_features(images)
        
        B, N_view, T = features.shape[:3]
        features = rearrange(features, 'b n t c h w -> (b n t) c h w')
        features = self.cosmos_in_proj(features)
        features = self.downsample_conv(features) 
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        self.cosmos_feat_shape = features.shape

        B, N_view, T, _, H, W = features.shape
        embs = rearrange(features, 'b n t c h w -> b (n t h w) c', b=B, n=N_view, t=T)
        # pad_masks = torch.ones((B, embs.shape[1]), dtype=torch.bool, device=device)
        pad_masks = torch.zeros((B, N_view, T, H, W), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, 'b n t h w -> b (n t h w)', b=B, n=N_view, t=T)

        att_masks = [1] + [0] * (embs.shape[1] - 1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(B, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_hidden_dim,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        if self.split_arm_hand_action:
            # Split actions into arm/hand token streams, then fuse each with timestep using a shared MLP.
            noisy_actions_arm, noisy_actions_hand = self._split_action_parts(noisy_actions)

            def arm_proj_func(arm_actions):
                return self.arm_action_in_proj(arm_actions)

            def hand_proj_func(hand_actions):
                return self.hand_action_in_proj(hand_actions)

            arm_action_emb = self._apply_checkpoint(arm_proj_func, noisy_actions_arm)
            hand_action_emb = self._apply_checkpoint(hand_proj_func, noisy_actions_hand)

            arm_time_emb = time_emb[:, None, :].expand_as(arm_action_emb)
            hand_time_emb = time_emb[:, None, :].expand_as(hand_action_emb)
            arm_action_time_emb = torch.cat([arm_action_emb, arm_time_emb], dim=2)
            hand_action_time_emb = torch.cat([hand_action_emb, hand_time_emb], dim=2)

            def mlp_func(action_time_input):
                x = self.action_time_mlp_in(action_time_input)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            arm_action_time_emb = self._apply_checkpoint(mlp_func, arm_action_time_emb)
            hand_action_time_emb = self._apply_checkpoint(mlp_func, hand_action_time_emb)

            embs.append(arm_action_time_emb)
            embs.append(hand_action_time_emb)
            bsize, arm_action_time_dim = arm_action_time_emb.shape[:2]
            _, hand_action_time_dim = hand_action_time_emb.shape[:2]
            arm_action_time_mask = torch.ones(bsize, arm_action_time_dim, dtype=torch.bool, device=timestep.device)
            hand_action_time_mask = torch.ones(bsize, hand_action_time_dim, dtype=torch.bool, device=timestep.device)
            pad_masks.append(arm_action_time_mask)
            pad_masks.append(hand_action_time_mask)

            # Set attention masks so that image, language and state inputs do not attend to action tokens
            total_action_tokens = arm_action_time_dim + hand_action_time_dim
            att_masks += [1] + ([0] * (total_action_tokens - 1))
        else:
            # Original single-stream behavior: one sequence of action tokens.
            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
            time_emb_full = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb_full], dim=2)

            def mlp_func_single(action_time_input):
                x = self.action_time_mlp_in(action_time_input)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func_single, action_time_emb)
            embs.append(action_time_emb)
            bsize, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
            pad_masks.append(action_time_mask)
            att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks): 
        L = lang_tokens.shape[1]
        pseudo_avail_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_avail_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        position_ids, rope_deltas = self.qwen3_vl_with_expert.und_expert.model.get_rope_index(
            padded_lang_tokens, 
            image_grid_thw, 
            attention_mask=attention_mask, 
        )
        return position_ids, rope_deltas
    
    def decode_cosmos(self, features):
        b, n, t, c, h, w = self.cosmos_feat_shape
        features = rearrange(features, 'b (n t h w) c -> b n t c h w', b=b, n=n, t=t, h=h, w=w)
        features = features.mean(2)  # b n c h w
        features = rearrange(features, 'b n c h w -> (b n) c h w')

        features = self.upsample_conv(features) # dtype: torch.float32
        h_upsampled, w_upsampled = features.shape[-2:]
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(b * n, -1, c)
        features = self.cosmos_out_proj(self.cosmos_out_layer_norm(features)) # dtype: torch.float32
        features = features.view(b, n, h_upsampled, w_upsampled, features.shape[-1])
        features = features.permute(0, 1, 4, 2, 3)
        return features
    
    def forward(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        actions,
        tactile_images,
        tactile_flow=None,
        tactile_raw=None,
        force_future=None,
        noise=None,
        time=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,  # remove the future observation
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        # Tactile generation input (flow or deform pair) -> per-finger tokens appended to middle_embs
        num_tactile_tokens = 0
        if self.config.use_tactile_gen_branch:
            if self.config.use_tactile_force:
                flow_emb = self.tactile_force_encoder(tactile_flow.to(dtype=torch.float32))  # (B, 10, hidden)
            else:
                flow_emb = self.tactile_flow_encoder(tactile_flow.to(dtype=torch.float32))  # (B, 10, hidden)
            tactile_tokens = self.tactile_token_proj(flow_emb)  # (B, 10, gen_proj_dim)
            num_tactile_tokens = tactile_tokens.shape[1]  # 10
            B, L_img, D = middle_embs.shape
            device = middle_embs.device
            middle_embs = torch.cat([middle_embs, tactile_tokens.to(dtype=middle_embs.dtype)], dim=1)
            middle_pad_masks = torch.cat(
                [middle_pad_masks, torch.ones(B, num_tactile_tokens, dtype=torch.bool, device=device)],
                dim=1,
            )
            middle_att_masks = torch.cat(
                [middle_att_masks, torch.zeros(B, num_tactile_tokens, dtype=torch.bool, device=device)],
                dim=1,
            )

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        tactile_out = self.tactile_encoder(tactile_images[:, :, 1])
        local_tactile_tokens = tactile_out["local_tokens"]
        global_tactile_token = tactile_out["global_token"]
        tac_dtype = next(self.qwen3_vl_with_expert.tactile_cross_attn_layers[0].parameters()).dtype
        local_tactile_tokens = local_tactile_tokens.to(dtype=tac_dtype)
        global_tactile_token = global_tactile_token.to(dtype=tac_dtype)

        def forward_func(prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids, local_tactile_tokens, global_tactile_token):
            (_, middle_out, suffix_out), _ = self.qwen3_vl_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                use_cache=False,
                local_tactile_tokens=local_tactile_tokens,
                global_tactile_token=global_tactile_token,
            )
            return middle_out, suffix_out

        middle_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids, local_tactile_tokens, global_tactile_token
        )

        # Split: image tokens -> visual latent (cosmos), tactile tokens -> tactile latent
        b, n, t, c, h, w = self.cosmos_feat_shape
        L_img = n * t * h * w
        image_out = middle_out[:, :L_img, :]
        if self.config.use_tactile_gen_branch:
            tactile_out = middle_out[:, L_img:, :]
        else:
            tactile_out = None

        def cosmos_out_func(image_out):
            return self.decode_cosmos(image_out)

        if self.config.use_visual_gen:
            pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, image_out.to(dtype=torch.float32))
            future_embs = self.get_cosmos_features(images[:, :, 2])
            loss_gen = F.mse_loss(pred_cosmos_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])
        else:
            loss_gen = torch.tensor(0.0, device=middle_out.device, dtype=torch.float32)

        if self.split_arm_hand_action:
            suffix_out = suffix_out[:, -2 * self.config.chunk_size :]
            arm_suffix_out = suffix_out[:, : self.config.chunk_size].to(dtype=torch.float32)
            hand_suffix_out = suffix_out[:, self.config.chunk_size :].to(dtype=torch.float32)

            def action_out_proj_func(arm_hidden, hand_hidden):
                return self._decode_action_parts(arm_hidden, hand_hidden)

            v_t = self._apply_checkpoint(action_out_proj_func, arm_suffix_out, hand_suffix_out)
        else:
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)

            def action_out_proj_func_single(suffix_hidden):
                return self.action_out_proj(suffix_hidden)

            v_t = self._apply_checkpoint(action_out_proj_func_single, suffix_out)

        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        # Tactile latent -> 3 decoders (deform, orig, 6D force); tactile latent from tactile tokens only
        loss_tactile = torch.tensor(0.0, device=loss_action.device, dtype=torch.float32)
        if self.config.use_tactile_gen_branch:
            tactile_latent = self.tactile_latent_proj(tactile_out.to(dtype=torch.float32))

            pred_deform = self.tactile_deform_decoder(tactile_latent)
            gt_deform = tactile_images[:, :, 2].to(dtype=torch.float32)
            if gt_deform.shape[-2:] != (224, 224):
                b, f, c, h, w = gt_deform.shape
                gt_deform = F.interpolate(
                    gt_deform.view(b * f, c, h, w), size=(224, 224), mode="bilinear", align_corners=False
                ).view(b, f, c, 224, 224)
            if gt_deform.max() <= 1.0:
                gt_deform = gt_deform * 2 - 1
            loss_tactile_deform = F.mse_loss(pred_deform, gt_deform)
            loss_tactile = loss_tactile + self.config.lambda_tactile_deform * loss_tactile_deform

            if tactile_raw is not None:
                pred_orig = self.tactile_orig_decoder(tactile_latent)
                gt_orig = tactile_raw.to(dtype=torch.float32)
                if gt_orig.ndim == 6:
                    gt_orig = gt_orig[:, :, -1]
                if gt_orig.shape[-2:] != (224, 224):
                    b, f, c, h, w = gt_orig.shape
                    gt_orig = F.interpolate(
                        gt_orig.view(b * f, c, h, w), size=(224, 224), mode="bilinear", align_corners=False
                    ).view(b, f, c, 224, 224)
                if gt_orig.max() <= 1.0:
                    gt_orig = gt_orig * 2 - 1
                loss_tactile_orig = F.mse_loss(pred_orig, gt_orig)
                loss_tactile = loss_tactile + self.config.lambda_tactile_orig * loss_tactile_orig

            if force_future is not None:
                pred_force = self.force_6d_decoder(tactile_latent)
                loss_force_future = F.mse_loss(pred_force, force_future)
                loss_tactile = loss_tactile + self.config.lambda_force_6d * loss_force_future
            
        # Auxiliary: supervise MultiFingerTactileEncoder by decoding local_tokens -> deform (same frame as encoder input, t=1)
        if self.config.lambda_tactile_encoder_deform > 0:
            pred_enc_def = self.tactile_encoder_deform_decoder(local_tactile_tokens.to(dtype=torch.float32))
            gt_enc_def = tactile_images[:, :, 1].to(dtype=torch.float32)
            if gt_enc_def.shape[-2:] != (224, 224):
                b, f, c, h, w = gt_enc_def.shape
                gt_enc_def = F.interpolate(
                    gt_enc_def.view(b * f, c, h, w), size=(224, 224), mode="bilinear", align_corners=False
                ).view(b, f, c, 224, 224)
            if gt_enc_def.max() <= 1.0:
                gt_enc_def = gt_enc_def * 2 - 1
            loss_tactile_enc_def = F.mse_loss(pred_enc_def, gt_enc_def)
            loss_tactile = loss_tactile + self.config.lambda_tactile_encoder_deform * loss_tactile_enc_def

        return loss_action, loss_gen, loss_tactile

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, tactile_images=None, tactile_flow=None, noise=None, num_steps=None, decode_image=False, decode_tactile=False,
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            # Sample noise with full action dimension; arm/hand split happens in embed_suffix.
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        # Tactile: same as forward — encode once, pass to all expert forwards
        local_tactile_tokens = None
        global_tactile_token = None
        if tactile_images is not None:
            tactile_out = self.tactile_encoder(tactile_images[:, :, 1])
            tac_dtype = next(self.qwen3_vl_with_expert.tactile_cross_attn_layers[0].parameters()).dtype
            local_tactile_tokens = tactile_out["local_tokens"].to(dtype=tac_dtype)
            global_tactile_token = tactile_out["global_token"].to(dtype=tac_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.qwen3_vl_with_expert.und_expert.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
            local_tactile_tokens=local_tactile_tokens,
            global_tactile_token=global_tactile_token,
        )
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,
        )

        num_tactile_tokens = 0
        if self.config.use_tactile_gen_branch:
            B, L_img, D = middle_embs.shape
            device = middle_embs.device
            dtype_mid = middle_embs.dtype
            if self.config.use_tactile_force:
                flow_emb = self.tactile_force_encoder(tactile_flow.to(dtype=torch.float32))  # (B, 10, hidden)
            else:
                flow_emb = self.tactile_flow_encoder(tactile_flow.to(dtype=torch.float32))  # (B, 10, hidden)
            tactile_tokens = self.tactile_token_proj(flow_emb).to(dtype=dtype_mid)  # (B, 10, gen_proj_dim)
            num_tactile_tokens = tactile_tokens.shape[1]  # 10
            B, L_img, D = middle_embs.shape
            device = middle_embs.device
            middle_embs = torch.cat([middle_embs, tactile_tokens.to(dtype=middle_embs.dtype)], dim=1)
            middle_pad_masks = torch.cat(
                [middle_pad_masks, torch.ones(B, num_tactile_tokens, dtype=torch.bool, device=device)],
                dim=1,
            )
            middle_att_masks = torch.cat(
                [middle_att_masks, torch.zeros(B, num_tactile_tokens, dtype=torch.bool, device=device)],
                dim=1,
            )

        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        middle_position_ids = torch.arange(1, middle_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_vl_with_expert.gen_expert.config._attn_implementation = "eager"  # noqa: SLF001

        (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs, None],
            use_cache=True,
            local_tactile_tokens=local_tactile_tokens,
            global_tactile_token=global_tactile_token,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                curr_pad_masks,
                past_key_values,
                max_position_ids, 
                x_t.to(dtype),
                expanded_time.to(dtype),
                local_tactile_tokens=local_tactile_tokens,
                global_tactile_token=global_tactile_token,
            )
            x_t = x_t + dt * v_t
            time += dt

        if decode_image:
            b, n, t, c, h, w = self.cosmos_feat_shape
            L_img = n * t * h * w
            image_out = middle_out[:, :L_img, :]
            def cosmos_out_func(image_out):
                return self.decode_cosmos(image_out)
            pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, image_out.to(dtype=torch.float32))
            # pred_cosmos_features = pred_cosmos_features.squeeze(0)
            recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
        else:
            recon_images = None

        if decode_tactile and num_tactile_tokens > 0 and self.tactile_latent_proj is not None:
            b, n, t, c, h, w = self.cosmos_feat_shape
            L_img = n * t * h * w
            tactile_out = middle_out[:, L_img:, :] # (B, 10, D)
            tactile_latent = self.tactile_latent_proj(tactile_out.to(dtype=torch.float32))
            recon_tactile = self.tactile_deform_decoder(tactile_latent)
            recon_tactile_orig = (
                self.tactile_orig_decoder(tactile_latent)
                if self.tactile_orig_decoder is not None
                else None
            )
            recon_force = self.force_6d_decoder(tactile_latent)
        else:
            recon_tactile = None
            recon_tactile_orig = None
            recon_force = None

        return x_t, recon_images, recon_tactile, recon_force, recon_tactile_orig

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids, 
        x_t,
        timestep,
        local_tactile_tokens=None,
        global_tactile_token=None,  # unused; kept for call-site compatibility
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        position_ids = torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_vl_with_expert.act_expert.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
            local_tactile_tokens=local_tactile_tokens,
        )

        suffix_out = outputs_embeds[2]
        if self.split_arm_hand_action:
            suffix_out = suffix_out[:, -2 * self.config.chunk_size :]
            arm_suffix_out = suffix_out[:, : self.config.chunk_size].to(dtype=torch.float32)
            hand_suffix_out = suffix_out[:, self.config.chunk_size :].to(dtype=torch.float32)
            return self._decode_action_parts(arm_suffix_out, hand_suffix_out)
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class DeCALPolicy(PreTrainedPolicy):
    """DeCAL tactile policy for LeRobot."""

    config_class = DeCALConfig
    name = "decal"

    def __init__(
        self,
        config: DeCALConfig,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = DeCALModel(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def __str__(self) -> str:
        lines = []

        # ---- basic info ----s
        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append("")

        # ---- parameter counts ----
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        num_und = sum(p.numel() for p in self.model.qwen3_vl_with_expert.und_expert.parameters())
        num_gen = sum(p.numel() for p in self.model.qwen3_vl_with_expert.gen_expert.parameters())
        num_act = sum(p.numel() for p in self.model.qwen3_vl_with_expert.act_expert.parameters())

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - Und params          : {num_und} ({format_big_number(num_und)})")
        lines.append(f"  - Gen params          : {num_gen} ({format_big_number(num_gen)})")
        lines.append(f"  - Act params          : {num_act} ({format_big_number(num_act)})")

        lines.append("=" * 60)

        return "\n".join(lines)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.cosmos.to(torch.bfloat16)
        if self.model.split_arm_hand_action:
            self.model.arm_action_out_proj.to(torch.float32)
            self.model.hand_action_out_proj.to(torch.float32)
        else:
            self.model.action_out_proj.to(torch.float32)
        return self

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.
        """
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)
        
        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks
    
    def _preprocess_tactile_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Preprocess tactile images for the model.
        """
        tactile_images_left_thumb_deform = batch[f"{OBS_TACTILE_IMAGES}.left_thumb_deform"]
        tactile_images_left_index_deform = batch[f"{OBS_TACTILE_IMAGES}.left_index_deform"]
        tactile_images_left_middle_deform = batch[f"{OBS_TACTILE_IMAGES}.left_middle_deform"]
        tactile_images_left_ring_deform = batch[f"{OBS_TACTILE_IMAGES}.left_ring_deform"]
        tactile_images_left_pinky_deform = batch[f"{OBS_TACTILE_IMAGES}.left_pinky_deform"]
        tactile_images_right_thumb_deform = batch[f"{OBS_TACTILE_IMAGES}.right_thumb_deform"]
        tactile_images_right_index_deform = batch[f"{OBS_TACTILE_IMAGES}.right_index_deform"]
        tactile_images_right_middle_deform = batch[f"{OBS_TACTILE_IMAGES}.right_middle_deform"]
        tactile_images_right_ring_deform = batch[f"{OBS_TACTILE_IMAGES}.right_ring_deform"]
        tactile_images_right_pinky_deform = batch[f"{OBS_TACTILE_IMAGES}.right_pinky_deform"]

        # concat all tactile images
        tactile_images = torch.stack([
            tactile_images_left_thumb_deform,
            tactile_images_left_index_deform,
            tactile_images_left_middle_deform,
            tactile_images_left_ring_deform,
            tactile_images_left_pinky_deform,
            tactile_images_right_thumb_deform,
            tactile_images_right_index_deform,
            tactile_images_right_middle_deform,
            tactile_images_right_ring_deform,
            tactile_images_right_pinky_deform
        ], dim=1)
        return tactile_images

    def _preprocess_tactile_force(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Return (prev_force, curr_force, future_force), each with shape (B, 10, 6)."""
        force_key = "observation.tactile"
        force = batch[force_key].to(dtype=torch.float32)
        # Temporal order from delta_timestamps: [-15, 0, +15].
        if force.shape[1] < 3:
            raise ValueError(f"Expected tactile force with 3 frames, got shape={tuple(force.shape)}")
        prev_force = force[:, 0].reshape(force.shape[0], 10, 6)
        curr_force = force[:, 1].reshape(force.shape[0], 10, 6)
        future_force = force[:, 2].reshape(force.shape[0], 10, 6)
        return prev_force, curr_force, future_force

    def _preprocess_tactile_gen_input(
        self,
        tactile_images: Tensor,
        tactile_force_prev: Tensor | None = None,
        tactile_force_curr: Tensor | None = None,
    ) -> Tensor | None:
        """Prepare tactile input for generation branch.
        - use_tactile_deform=False: optical flow from two deform frames, shape (B, 10, 2, H, W).
        - use_tactile_deform=True: concatenate deform (t-1, t) along channel, shape (B, 10, 6, H, W).
        - use_tactile_force=True: force-only input from history/current force, shape (B, 10, 12).
        """
        if tactile_images.shape[2] < 2 and not self.config.use_tactile_force:
            return None
        if self.config.use_tactile_force:
            if tactile_force_prev is None or tactile_force_curr is None:
                raise ValueError("use_tactile_force=True but tactile force is missing from batch.")
            return torch.cat([tactile_force_prev, tactile_force_curr], dim=2)
        if self.config.use_tactile_deform:
            tactile_prev = tactile_images[:, :, 0].to(dtype=torch.float32)
            tactile_curr = tactile_images[:, :, 1].to(dtype=torch.float32)
            return torch.cat([tactile_prev, tactile_curr], dim=2)

        tactile_flow = compute_tactile_flow(tactile_images, t_prev=0, t_curr=1)  # Causal history-to-current flow.
        return tactile_flow / self.config.tactile_flow_scale

    def _preprocess_tactile_raw(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Preprocess tactile raw images (B, 10, T, C, H, W). Returns None if not in batch."""
        tactile_images_left_thumb_raw = batch[f"{OBS_TACTILE_IMAGES}.left_thumb_raw"]
        tactile_images_left_index_raw = batch[f"{OBS_TACTILE_IMAGES}.left_index_raw"]
        tactile_images_left_middle_raw = batch[f"{OBS_TACTILE_IMAGES}.left_middle_raw"]
        tactile_images_left_ring_raw = batch[f"{OBS_TACTILE_IMAGES}.left_ring_raw"]
        tactile_images_left_pinky_raw = batch[f"{OBS_TACTILE_IMAGES}.left_pinky_raw"]
        tactile_images_right_thumb_raw = batch[f"{OBS_TACTILE_IMAGES}.right_thumb_raw"]
        tactile_images_right_index_raw = batch[f"{OBS_TACTILE_IMAGES}.right_index_raw"]
        tactile_images_right_middle_raw = batch[f"{OBS_TACTILE_IMAGES}.right_middle_raw"]
        tactile_images_right_ring_raw = batch[f"{OBS_TACTILE_IMAGES}.right_ring_raw"]
        tactile_images_right_pinky_raw = batch[f"{OBS_TACTILE_IMAGES}.right_pinky_raw"]

        # concat all tactile images
        tactile_images = torch.stack([
            tactile_images_left_thumb_raw,
            tactile_images_left_index_raw,
            tactile_images_left_middle_raw,
            tactile_images_left_ring_raw,
            tactile_images_left_pinky_raw,
            tactile_images_right_thumb_raw,
            tactile_images_right_index_raw,
            tactile_images_right_middle_raw,
            tactile_images_right_ring_raw,
            tactile_images_right_pinky_raw
        ], dim=1)
        return tactile_images

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def prepare_gen_features(self, batch):
        images = torch.stack([batch[f"{OBS_IMAGES}.image{i}"] for i in range(3)], dim=1)  # B, N_view, T, C, H, W
        B, N_view, T = images.shape[:3]
        images = rearrange(images, 'b n t c h w -> (b n t) c h w')
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.model.cosmos.encode(images)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        return features

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions, *_ = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], decode_image=False, decode_tactile=False) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        images, img_masks = self._preprocess_images(batch)
        tactile_images = self._preprocess_tactile_images(batch)
        force_prev, force_curr, _ = self._preprocess_tactile_force(batch)
        tactile_flow = self._preprocess_tactile_gen_input(tactile_images, force_prev, force_curr)

        # Sample actions using the model
        actions, recon_images, recon_tactile, recon_force, recon_tactile_orig = self.model.sample_actions(
            images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
            tactile_images=tactile_images, tactile_flow=tactile_flow,
            decode_image=decode_image, decode_tactile=decode_tactile,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, recon_images, recon_tactile, recon_force, recon_tactile_orig

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]

        images, img_masks = self._preprocess_images(batch)
        tactile_images = self._preprocess_tactile_images(batch)

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        tactile_raw = self._preprocess_tactile_raw(batch)
        force_prev, force_curr, force_future = self._preprocess_tactile_force(batch)
        tactile_flow = self._preprocess_tactile_gen_input(tactile_images, force_prev, force_curr)

        result = self.model.forward(
            images, img_masks, pixel_values, image_grid_thw,
            lang_tokens, lang_masks, state, actions, tactile_images,
            tactile_flow=tactile_flow, tactile_raw=tactile_raw, force_future=force_future,
        )
        if len(result) == 3:
            losses_action, loss_gen, loss_tactile = result
        else:
            losses_action, loss_gen = result
            loss_tactile = torch.tensor(0.0, device=losses_action.device)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses_action = losses_action[:, :, :original_action_dim]

        if self.config.use_consistency_action_loss:
            # Compute chunk consistency from GT actions: c = exp(-alpha * R),
            # where R = Ej / (Ev + eps), Ej from 2nd-order diff, Ev from 1st-order diff.
            gt_action = actions[:, :, :original_action_dim].to(dtype=torch.float32)
            if gt_action.shape[1] >= 3:
                vel = gt_action[:, 1:, :] - gt_action[:, :-1, :]
                jerk = vel[:, 1:, :] - vel[:, :-1, :]
                e_v = vel.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
                e_j = jerk.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
                r = e_j / (e_v + self.config.consistency_eps)
                c = torch.exp(-self.config.consistency_alpha * r)
            else:
                c = torch.ones(gt_action.shape[0], device=gt_action.device, dtype=gt_action.dtype)

            threshold = max(float(self.config.consistency_threshold), 1e-6)
            weights = torch.where(c > threshold, torch.ones_like(c), c / threshold).clamp(min=0.0, max=1.0)
            losses_action = losses_action * weights[:, None, None].to(dtype=losses_action.dtype)

        loss_action = losses_action.mean()

        loss = loss_action + self.config.lambda_gen * loss_gen + loss_tactile

        loss_dict = {
            "loss": loss.item(),
            "loss_action": loss_action.item(),
            "loss_gen": loss_gen.item(),
            "loss_tactile": loss_tactile.item(),
        }
        
        losses_action = losses_action.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        loss_dict.update({
            f"loss_action_dim{i}": losses_action[i] for i in range(original_action_dim)
        })

        return loss, loss_dict


if __name__ == "__main__":
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION
    from lerobot.policies.DeCAL.transform_decal import DeCALProcessorTransformFn
    from pprint import pp
    torch.manual_seed(0)
    device = torch.device("cuda")

    processor = DeCALProcessorTransformFn()

    cfg = DeCALConfig()
    cfg.qwen3_vl_variant="qwen3_vl_28l"
    cfg.action_expert_variant="qwen3_28l"
    cfg.freeze_vision_encoder=True
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.bfloat16

    model = DeCALPolicy(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}  ({trainable_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")

    B = 2
    samples = [{
        f"{OBS_IMAGES}.image0": torch.rand((3, 3, 224, 224)), 
        f"{OBS_IMAGES}.image1": torch.rand((3, 3, 224, 224)), 
        f"{OBS_IMAGES}.image2": torch.rand((3, 3, 224, 224)), 
        f"{OBS_IMAGES}.image0_mask": torch.tensor(True).cuda(), 
        f"{OBS_IMAGES}.image1_mask": torch.tensor(True).cuda(), 
        f"{OBS_IMAGES}.image2_mask": torch.tensor(True).cuda(), 
        "task": f"This is test sample {i}.", 
        OBS_STATE: torch.rand((14, )), 
        ACTION: torch.rand((50, 14)), 
    } for i in range(B)]
    samples = [processor(sample) for sample in samples]
    inputs = {}
    for key in samples[0].keys():
        if key != "task":
            inputs[key] = torch.stack([sample[key] for sample in samples], dim=0).to(device=device)
    loss, loss_dict = model.forward(inputs)
    pp(loss)
    pp(loss_dict)
    
