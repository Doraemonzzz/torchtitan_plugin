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


@dataclass
class ParallelDims:
    dp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, tp, pp = self.dp, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (tp * pp)
        assert dp >= 1, dp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp * tp * pp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp, self.tp], ["pp", "dp", "tp"], strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        logging_info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

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
    def model_parallel_size(self):
        return self.tp * self.pp
