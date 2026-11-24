"""Load and run a native AGM 6 checkpoint with PyTorch and SafeTensors."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


class AGM6:
    def __init__(self, directory, device):
        from safetensors import safe_open
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.config = json.loads((directory / 'agm-config.json').read_text(encoding='utf-8'))
        if self.config['format'] != 'agm-6-v1':
            raise ValueError('Expected an AGM 6 native checkpoint')
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        files = sorted(directory.glob('model-*-of-*.safetensors'))
        if not files:
            raise FileNotFoundError('Model shards are missing. Run git lfs pull.')
        self.weights = {}
        for path in files:
            with safe_open(path, framework='pt', device='cpu') as archive:
                for key in archive.keys():
                    self.weights[key] = archive.get_tensor(key).to(device, copy=True)
        self.embedding = self.weights['model.embed_tokens.weight']
        if self.embedding.shape != (self.tokenizer.get_vocab_size(), self.config['width']):
            raise ValueError('Tokenizer and embedding dimensions do not match')
        self.device = self.embedding.device
        dim = self.config['head_dim']
        rotary = self.config['rotary']
        inv = 1 / rotary['base'] ** (torch.arange(0, dim, 2, device=self.device).float() / dim)
        wavelength = 2 * math.pi / inv
        shorter = rotary['original_context'] / rotary['high_frequency']
        longer = rotary['original_context'] / rotary['low_frequency']
        fraction = (rotary['original_context'] / wavelength - rotary['low_frequency'])
        fraction = (fraction / (rotary['high_frequency'] - rotary['low_frequency'])).clamp(0, 1)
        self.inv_freq = torch.where(
            wavelength < shorter, inv,
            torch.where(wavelength > longer, inv / rotary['scale'],
                        (1 - fraction) * inv / rotary['scale'] + fraction * inv),
        )

    def norm(self, x, key):
        full = x.float()
        normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + self.config['norm_epsilon'])
        return self.weights[key] * normalized.to(x.dtype)

    def format_chat(self, messages):
        chat = self.config['chat']
        messages = list(messages)
        if not messages or messages[0]['role'] != 'system':
            messages.insert(0, dict(role='system', content=chat['default_system']))
        text = chat['bos']
        for message in messages:
            role = message['role']
            if role not in ('system', 'user', 'assistant'):
                raise ValueError(f'Unsupported role: {role}')
            text += (chat['header_open'] + role + chat['header_close'] + ' ' +
                     message['content'].strip() + chat['turn_end'])
        return text + chat['header_open'] + 'assistant' + chat['header_close'] + ' '

    @torch.inference_mode()
    def forward(self, ids, cache=None, feature_positions=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected a nonempty batch of token IDs')
        if cache is not None and ids.shape[1] != 1:
            raise ValueError('Cached decoding expects exactly one new token')
        length = ids.shape[1]
        offset = cache[0][0].shape[-2] if cache is not None else 0
        if offset + length > self.config['context']:
            raise ValueError('Input exceeds the checkpoint context')
        periods = torch.outer(torch.arange(offset, offset + length, device=self.device).float(), self.inv_freq)
        angles = torch.cat((periods, periods), dim=-1)
        cos = angles.cos().to(self.embedding.dtype)
        sin = angles.sin().to(self.embedding.dtype)
        x = F.embedding(ids, self.embedding)
        new_cache = []
        dim = self.config['head_dim']
        groups = self.config['heads'] // self.config['kv_heads']

        def rotate(value):
            first, second = value.chunk(2, dim=-1)
            return value * cos + torch.cat((-second, first), dim=-1) * sin

        for index in range(self.config['layers']):
            block = f'model.layers.{index}.'
            h = self.norm(x, block + 'input_layernorm.weight')

            def project(name):
                value = F.linear(h, self.weights[block + f'self_attn.{name}_proj.weight'])
                return value.reshape(ids.shape[0], length, -1, dim).transpose(1, 2)

            q, k, v = rotate(project('q')), rotate(project('k')), project('v')
            if cache is not None:
                k = torch.cat((cache[index][0], k), dim=-2)
                v = torch.cat((cache[index][1], v), dim=-2)
            new_cache.append((k, v))
            attended = F.scaled_dot_product_attention(
                q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1),
                is_causal=cache is None,
            )
            attended = attended.transpose(1, 2).reshape_as(x)
            x = x + F.linear(attended, self.weights[block + 'self_attn.o_proj.weight'])
            h = self.norm(x, block + 'post_attention_layernorm.weight')
            gate = F.silu(F.linear(h, self.weights[block + 'mlp.gate_proj.weight']))
            up = F.linear(h, self.weights[block + 'mlp.up_proj.weight'])
            x = x + F.linear(gate * up, self.weights[block + 'mlp.down_proj.weight'])
        if feature_positions is not None:
            return self.norm(x[torch.arange(ids.shape[0], device=self.device), feature_positions],
                             'model.norm.weight')
        x = self.norm(x[:, -1:], 'model.norm.weight')
        return F.linear(x, self.embedding).float(), new_cache

    def generate(self, messages, max_tokens):
        ids = self.tokenizer.encode(self.format_chat(messages), add_special_tokens=False).ids
        tokens = torch.tensor([ids], device=self.device)
        cache, result = None, []
        for _ in range(max_tokens):
            logits, cache = self.forward(tokens, cache)
            tokens = logits.argmax(dim=-1)
            token = tokens.item()
            if token in self.config['stop_ids']:
                break
            result.append(token)
        return self.tokenizer.decode(result, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    directory = Path(__file__).resolve().parent
    parser.add_argument('--model', type=Path, default=directory / 'models/agm-6')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tokens', type=int, default=64)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('--tokens must be positive')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AGM6(args.model, args.device)
    print(model.generate([dict(role='user', content=args.prompt)], args.tokens))


if __name__ == '__main__':
    main()
