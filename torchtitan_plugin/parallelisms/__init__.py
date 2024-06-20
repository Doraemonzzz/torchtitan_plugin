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
