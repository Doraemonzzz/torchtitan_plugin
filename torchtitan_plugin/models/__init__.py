from torchtitan.models.llama import Transformer
from .model_llama import llama2_configs, llama3_configs

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
}

model_name_to_cls = {"llama2": Transformer, "llama3": Transformer}
