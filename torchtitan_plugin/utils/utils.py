
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        if hasattr(model, "tok_embeddings"):
            num_params -= model.tok_embeddings.weight.numel()
        else:
            num_params -= model.model.embed_tokens.weight.numel()
    return num_params


# update this later
def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    try:
        l, h, q, t = (
            model_config.n_layers,
            model_config.n_heads,
            model_config.dim // model_config.n_heads,
            seq_len,
        )
    except:
        l, h, q, t = (
            model_config.num_layers,
            model_config.num_heads,
            model_config.embed_dim // model_config.num_heads,
            seq_len,
        )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token
