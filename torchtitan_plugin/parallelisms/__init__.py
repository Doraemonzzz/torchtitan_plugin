from .parallelize_llama import parallelize_llama, pipeline_llama

models_parallelize_fns = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
}

models_pipelining_fns = {
    "llama2": pipeline_llama,
    "llama3": pipeline_llama,
}
