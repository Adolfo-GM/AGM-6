"""AgmGPT transformer with PyTorch grouped-query attention."""
import torch
from torch import nn
from torch.nn import functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos + right * sin, right * cos - left * sin), dim=-1)


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads, value_embedding):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.c_q = Linear(width, width, bias=False)
        self.c_k = Linear(width, kv_heads * self.head_dim, bias=False)
        self.c_v = Linear(width, kv_heads * self.head_dim, bias=False)
        self.c_proj = Linear(width, width, bias=False)
        self.ve_gate = Linear(12, kv_heads, bias=False) if value_embedding else None

    def forward(self, x, values, rotary, window, cache, position):
        batch, length, width = x.shape
        q = self.c_q(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.c_k(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.c_v(x).reshape_as(k)
        if values is not None:
            gate = 3 * self.ve_gate(x[..., :12]).sigmoid()
            v = v + gate.unsqueeze(-1) * values.reshape_as(v)

        q = 1.2 * norm(rotate(q, *rotary))
        k = 1.2 * norm(rotate(k, *rotary))
        end = position + length
        if cache is not None:
            keys, vals = cache
            keys[:, position:end].copy_(k)
            vals[:, position:end].copy_(v)
            k, v = keys[:, :end], vals[:, :end]

        mask = None
        causal = position == 0 and length <= window
        if length == 1:
            k, v = k[:, -(window + 1):], v[:, -(window + 1):]
            causal = False
        elif not causal:
            queries = torch.arange(position, end, device=x.device)[:, None]
            distance = queries - torch.arange(end, device=x.device)
            mask = (distance >= 0) & (distance <= window)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, is_causal=causal, enable_gqa=self.heads != self.kv_heads,
        )
        return self.c_proj(y.transpose(1, 2).reshape(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = Linear(width, 4 * width, bias=False)
        self.c_proj = Linear(4 * width, width, bias=False)

    def forward(self, x):
        return self.c_proj(self.c_fc(x).relu().square())


class DecoderLayer(nn.Module):
    def __init__(self, config, value_embedding):
        super().__init__()
        self.attn = Attention(config.n_embd, config.n_head, config.n_kv_head, value_embedding)
        self.mlp = FeedForward(config.n_embd)

    def forward(self, x, values, rotary, window, cache, position):
        x = x + self.attn(norm(x), values, rotary, window, cache, position)
        return x + self.mlp(norm(x))


class Decoder(nn.Module):
    """Parameter names and tensor shapes follow the shipped checkpoint schema."""

    def __init__(self, config, compute_dtype):
        super().__init__()
        self.config = config
        vocab = ((config.vocab_size + 63) // 64) * 64
        width, layers = config.n_embd, config.n_layer
        head_dim = width // config.n_head
        value_layers = range((layers - 1) % 2, layers, 2)
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab, width),
            'h': nn.ModuleList(DecoderLayer(config, i in value_layers)
                               for i in range(layers)),
        })
        self.lm_head = Linear(width, vocab, bias=False)
        self.smear_gate = Linear(24, 1, bias=False)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(vocab, config.n_kv_head * head_dim)
            for i in value_layers
        })
        self.resid_lambdas = nn.Parameter(torch.linspace(1.15, 1.05, layers))
        self.x0_lambdas = nn.Parameter(torch.linspace(.20, .05, layers))
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.full((1,), .2))

        short = -(-config.block_size // 4 // 128) * 128
        self.windows = [
            short if config.window_pattern[i % len(config.window_pattern)] == 'S'
            else config.block_size for i in range(layers)
        ]
        self.windows[-1] = config.block_size
        frequencies = 1.0 / (100000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(config.block_size, dtype=torch.float32), frequencies)
        self.register_buffer('cos', angles.cos()[None, :, None, :].to(compute_dtype), persistent=False)
        self.register_buffer('sin', angles.sin()[None, :, None, :].to(compute_dtype), persistent=False)

        init = nn.init
        bound = (3 / width) ** .5
        init.normal_(self.transformer.wte.weight, std=.8)
        init.normal_(self.lm_head.weight, std=.001)
        init.uniform_(self.smear_gate.weight, 0, .02)
        for embedding in self.value_embeds.values():
            init.uniform_(embedding.weight, -bound, bound)
        for layer in self.transformer.h:
            for projection in (layer.attn.c_q, layer.attn.c_k, layer.attn.c_v):
                init.uniform_(projection.weight, -bound, bound)
            init.zeros_(layer.attn.c_proj.weight)
            init.uniform_(layer.mlp.c_fc.weight, -.4 * bound, .4 * bound)
            init.zeros_(layer.mlp.c_proj.weight)
            if layer.attn.ve_gate is not None:
                init.uniform_(layer.attn.ve_gate.weight, 0, .02)

    def forward(self, idx, targets=None, kv_cache=None):
        length = idx.size(1)
        position = kv_cache.get_pos() if kv_cache is not None else 0
        end = position + length
        if end > self.config.block_size or (kv_cache is not None and end > kv_cache.k_cache.size(2)):
            raise ValueError('Input and cache exceed context window')
        rotary = self.cos[:, position:end], self.sin[:, position:end]
        x = norm(self.transformer.wte(idx).to(self.cos.dtype))
        previous = None
        if kv_cache is not None:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:]
        if previous is None:
            gate = self.smear_lambda.to(x.dtype) * self.smear_gate(x[:, 1:, :24]).sigmoid()
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        else:
            preceding = torch.cat((previous, x[:, :-1]), dim=1)
            gate = self.smear_lambda.to(x.dtype) * self.smear_gate(x[..., :24]).sigmoid()
            x = x + gate * preceding

        initial = x
        for i, layer in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * initial
            values = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            cache = kv_cache.get_layer_cache(i) if kv_cache is not None else None
            x = layer(x, values, rotary, self.windows[i], cache, position)
            if i == self.config.n_layer // 2:
                midpoint = x
        if kv_cache is not None:
            kv_cache.advance(length)
        x = x - self.backout_lambda.to(x.dtype) * midpoint
        logits = self.lm_head(norm(x))[..., :self.config.vocab_size].float()
        logits = 15 * (logits / 15).tanh()
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits
