from xmixers.models import LLaMAConfig

from .utils import get_glu_dim

llama_configs = {
    "debugmodel": LLaMAConfig(
        embed_dim=256,
        mid_dim=get_glu_dim(256),
        num_layers=8,
        num_heads=16,
        tie_word_embeddings=False,
    ),
    "410M": LLaMAConfig(
        embed_dim=1024,
        mid_dim=get_glu_dim(1024),
        num_layers=26,
        num_heads=8,
        tie_word_embeddings=False,
    ),
    "7B": LLaMAConfig(
        embed_dim=4096,
        mid_dim=get_glu_dim(4096),
        num_layers=32,
        num_heads=32,
        tie_word_embeddings=False,
    ),
}
