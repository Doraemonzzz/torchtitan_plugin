# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.


import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig

from torchtitan_plugin.utils import logging_info

from .utils import checkpoint_wrapper, get_tp_parallel_strategy


def apply_tp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism.
    """

    tp_mesh = world_mesh["tp"]
    (
        row_parallel_strategy,
        col_parallel_strategy,
        prepare_module_input,
    ) = get_tp_parallel_strategy(job_config)
    loss_parallel = parallel_dims.loss_parallel_enabled

    # 1. Parallelize the first embedding and the last linear proj layer
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Shard the first transformer block's inputs
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "lm_head": col_parallel_strategy(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "final_norm": SequenceParallel(),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    for layer_id, block in enumerate(model.model.layers):
        layer_plan = {
            "token_mixer": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "token_mixer.q_proj": col_parallel_strategy(),
            "token_mixer.k_proj": col_parallel_strategy(),
            "token_mixer.v_proj": col_parallel_strategy(),
            "token_mixer.out_proj": row_parallel_strategy(output_layouts=Shard(1)),
            "token_norm": SequenceParallel(),
            "channel_mixer": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "channel_mixer.w1": col_parallel_strategy(),
            "channel_mixer.w2": col_parallel_strategy(),
            "channel_mixer.w3": row_parallel_strategy(output_layouts=Shard(1)),
            "channel_norm": SequenceParallel(),
        }

        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logging_info("Applied Tensor Parallelism to the model")
    return model


def apply_ac(model, job_config: JobConfig):
    """
    Apply activation checkpointing to the model.
    """

    ac_config = job_config.activation_checkpoint

    for layer_id, block in enumerate(model.model.layers):
        block = checkpoint_wrapper(block, ac_config)
        model.model.layers[layer_id] = block

    logging_info(f"Applied {ac_config.mode} activation checkpointing to the model")
    return model


def apply_compile(model, job_config: JobConfig):
    """
    Apply torch.compile to the model.
    """

    if job_config.model.norm_type == "fused_rmsnorm":
        raise NotImplementedError(
            "fused_rmsnorm not yet compatible with torch.compile. Please use layernorm or rmsnorm."
        )

    for layer_id, block in model.model.layers.named_children():
        # turn on per-transformer block compile after AC wrapping and before FSDP
        # TODO: dynamic shape have some issues so we turn it off for now.
        # TODO: inline inbuilt nn modules does not work yet, enable it to accelarate
        # compile time.
        # torch._dynamo.config.inline_inbuilt_nn_modules = True
        block = torch.compile(block, dynamic=False)
        model.model.layers[layer_id] = block

    ac_config = job_config.activation_checkpoint
    if ac_config.mode == "selective" and ac_config.selective_ac_option == "op":
        # some temp flags for torch.compile enablement + SAC
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
            True
        )

    logging_info("Compiled each TransformerBlock with torch.compile")
    return model


def apply_dp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """

    dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
    assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, block in enumerate(model.model.layers):
        # for layer_id, block in model.model.layers.items():
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately.
        # When using Pipeline Parallelism, generally zero-2 is best so as to avoid repeated reshardings
        # per microbatch.
        reshard_after_forward = (
            int(layer_id) < len(model.model.layers) - 1 and not parallel_dims.pp_enabled
        )
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.model.layers[layer_id] = block

    model = fully_shard(
        model, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled
    )

    logging_info("Applied FSDP to the model")
    return model


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh, parallel_dims, job_config)

    if job_config.activation_checkpoint.mode != "none":
        model = apply_ac(model, job_config)

    if job_config.training.compile:
        model = apply_compile(model, job_config)

    if parallel_dims.dp_enabled:
        model = apply_dp(model, world_mesh, parallel_dims, job_config)

    return model


pipeline_llama = None
