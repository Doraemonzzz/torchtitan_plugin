from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh

from ..utils import logging_info
from .parallelize_llama import parallelize_llama, pipeline_llama
from .parallelize_llama_torchtitan import (
    parallelize_llama_torchtitan,
    pipeline_llama_torchtitan,
)

models_parallelize_fns = {
    "llama": parallelize_llama,
    "llama2": parallelize_llama_torchtitan,
    "llama3": parallelize_llama_torchtitan,
}

models_pipelining_fns = {
    "llama": pipeline_llama,
    "llama2": pipeline_llama_torchtitan,
    "llama3": pipeline_llama_torchtitan,
}

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging import logger


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (cp * tp * pp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert cp >= 1, cp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                if (name == "dp_replicate" and self.dp_shard == 1) or (
                    name == "dp_shard" and self.dp_replicate == 1
                ):
                    names.append("dp")
                else:
                    names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if self.dp_replicate > 1 and self.dp_shard > 1:
            mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

        if self.cp > 1:
            if self.dp_replicate > 1 and self.dp_shard > 1:
                mesh["dp_replicate", "dp_shard", "cp"]._flatten(mesh_dim_name="dp_cp")
            else:
                mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp
