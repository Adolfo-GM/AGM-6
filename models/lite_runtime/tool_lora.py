"""Attach frozen-base low-rank updates without merging the base weights."""
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from checkpoint import atomic_save, tokenizer_hash
from transformer import Linear

FORMAT = 'agmgpt-tool-lora-v1'


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.enabled = True
        self.down = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.up = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** .5)

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            update = F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype))
            output = output + self.scale * update
        return output


def attach(model, rank, alpha):
    if not isinstance(rank, int) or not 1 <= rank <= 64 or not 0 < alpha <= 128:
        raise ValueError('Invalid LoRA dimensions')
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('Model already has a LoRA adapter')
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, Linear)]
    model.requires_grad_(False)
    adapters = {}
    for name, layer in layers:
        parent, attr = name.rsplit('.', 1)
        adapter = LoRALinear(layer, rank, alpha)
        setattr(model.get_submodule(parent), attr, adapter)
        adapters[name] = adapter
    return adapters


@contextmanager
def adapter_enabled(adapters):
    previous = [layer.enabled for layer in adapters.values()]
    for layer in adapters.values():
        layer.enabled = True
    try:
        yield
    finally:
        for layer, enabled in zip(adapters.values(), previous):
            layer.enabled = enabled


def save_adapter(path, adapters, rank, alpha, base, tokenizer):
    weights = {
        name: {'down': layer.down.detach().cpu(), 'up': layer.up.detach().cpu()}
        for name, layer in adapters.items()
    }
    atomic_save(path, dict(format=FORMAT, rank=rank, alpha=alpha, weights=weights,
                           base_sha256=tokenizer_hash(base), tokenizer_sha256=tokenizer_hash(tokenizer)))


def load_adapter(model, path, base, tokenizer):
    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    return apply_adapter(model, saved, tokenizer_hash(base), tokenizer_hash(tokenizer))


def apply_adapter(model, saved, base_hash, vocab_hash):
    if (saved['format'] != FORMAT or saved['base_sha256'] != base_hash
            or saved['tokenizer_sha256'] != vocab_hash):
        raise ValueError('Tool adapter does not match the chat checkpoint or tokenizer')
    layers = {name: module for name, module in model.named_modules() if isinstance(module, Linear)}
    rank = saved['rank']
    weights = saved['weights']
    if weights.keys() != layers.keys():
        raise ValueError('Tool adapter layer names differ from the model')
    for name, layer in layers.items():
        down, up = weights[name]['down'], weights[name]['up']
        if (down.shape != (rank, layer.in_features) or up.shape != (layer.out_features, rank)
                or not down.is_floating_point() or not up.is_floating_point()
                or not down.isfinite().all() or not up.isfinite().all()):
            raise ValueError(f'Invalid tool adapter weights for {name}')
    adapters = attach(model, rank, saved['alpha'])
    with torch.no_grad():
        for name, layer in adapters.items():
            layer.down.copy_(weights[name]['down'])
            layer.up.copy_(weights[name]['up'])
            layer.enabled = False
            layer.requires_grad_(False)
    return adapters
