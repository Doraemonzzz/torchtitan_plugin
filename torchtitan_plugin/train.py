# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from typing import Any, Dict, List

import torch
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager
from torchtitan.float8 import Float8Handler
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

try:
    from torchtitan.parallelisms.pipelining_utils import build_pipeline_schedule
except:
    build_pipeline_schedule = None

from transformers import AutoModelForCausalLM

from torchtitan_plugin.data import build_data_loader
from torchtitan_plugin.models import model_name_to_cls, models_config
from torchtitan_plugin.parallelisms import (
    ParallelDims,
    models_parallelize_fns,
    models_pipelining_fns,
)
from torchtitan_plugin.tokenizer import create_tokenizer
from torchtitan_plugin.utils import (
    JobConfig,
    build_gpu_memory_monitor,
    build_metric_logger,
    get_num_flop_per_token,
    get_num_params,
    init_logger,
    logging_info,
    maybe_enable_profiling,
)


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizer(model, job_config: JobConfig):
    # build optimizer
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused

    # Common parameters for both optimizers
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    return optimizer


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logging_info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # set determinisism, use seed == None to skip deterministic training
    utils.set_determinism(job_config.training.seed)
    if job_config.training.seed is None:
        logging_info("Deterministic training off")
    else:
        logging_info(
            f"Deterministic training on. Using seed: {job_config.training.seed}"
        )

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)
    logging_info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = job_config.model.tokenizer_type
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # build model (using meta init)
    model_cls = model_name_to_cls.get(model_name, AutoModelForCausalLM)
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    logging_info(f"Building {model_name} {job_config.model.flavor} with {model_config}")

    with torch.device("meta"):
        try:
            logging_info(f"Using torchtitan model")
            model = model_cls.from_model_args(model_config)
        except:
            logging_info(f"Using xmixers model")
            model = model_cls.from_config(model_config)

    logging_info(model)

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logging_info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    if job_config.training.compile:
        loss_fn = torch.compile(loss_fn)

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        pp_schedule, model_parts = models_pipelining_fns[model_name](
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
            m.to_empty(device="cuda")
            m.init_weights()
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)

        # move sharded model to CPU/GPU and initialize weights via DTensor
        init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
        model.to_empty(device=init_device)
        model.init_weights()
        model.train()

        model_parts = [model]

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logging_info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logging_info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load()

    if parallel_dims.pp_enabled and not checkpoint_loaded:
        # TODO: fix this by allowing each rank to set their own seed
        logger.warning(
            "Pipeline Parallelism is being used without a seed checkpoint. "
            "All the substages will be initialized with random weights with same RNG state which can affect convergence."
        )

    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logging_info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            input_ids, labels = batch
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.cuda()
            labels = labels.cuda()
            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels, model.freqs_cis],
                    cp_seq_dims=[1, 1, 0],
                    cp_no_restore_buffers={input_ids, labels},
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                with train_context(optional_context_parallel_ctx):
                    if pp_mesh.get_local_rank() == 0:
                        pp_schedule.step(input_ids)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=labels, losses=losses)
                    else:
                        pp_schedule.step()

                # accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses))
                    if is_last_stage
                    else torch.Tensor([-1.0])
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    pred = model(input_ids)
                    if hasattr(pred, "logits"):
                        pred = pred.logits
                    loss = loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), job_config.training.max_norm, foreach=True
                )

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, dp_mesh),
                        utils.dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logging_info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}wps: {round(wps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logging_info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logging_info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    destroy_process_group()
