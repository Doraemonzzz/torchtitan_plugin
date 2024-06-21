from xmixers.models import LLaMAConfig

llama_configs = {
    "debugmodel": LLaMAConfig(embed_dim=256, num_layers=8, num_heads=16),
    "410M": LLaMAConfig(embed_dim=1024, num_layers=26, num_heads=8),
    "7B": LLaMAConfig(embed_dim=4096, num_layers=32, num_heads=32),
}
