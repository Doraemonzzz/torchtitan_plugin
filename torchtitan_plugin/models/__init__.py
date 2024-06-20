from torchtitan.models.llama import Transformer

from .model_llama import llama_configs
from .model_llama_torchtitan import llama2_configs, llama3_configs

models_config = {
    "llama": llama_configs,
    "llama2": llama2_configs,  # torchtitan
    "llama3": llama3_configs,  # torchtitan
}

model_name_to_cls = {"llama2": Transformer, "llama3": Transformer}
