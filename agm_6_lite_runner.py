"""Run AGM 6 Lite from its single model file."""
import argparse
import hashlib
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'models/lite_runtime'))

from tokenizers import Tokenizer

from chat import run_chat
from chat_tools import choose_tool, invoke, reply_with_result, user_content
from checkpoint import load_model
from tool_lora import adapter_enabled, apply_adapter


class LiteChat:
    def __init__(self, path, device):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') != 'agm-6-lite-v1':
            raise ValueError('Expected an AGM 6 Lite model file')
        checkpoint = saved.pop('checkpoint').numpy().tobytes()
        vocab = saved['tokenizer']
        self.base_hash = hashlib.sha256(checkpoint).hexdigest()
        self.vocab_hash = hashlib.sha256(vocab).hexdigest()
        with io.BytesIO(checkpoint) as stream:
            self.model = load_model(stream, device, self.vocab_hash)
        self.tokenizer = Tokenizer.from_str(vocab.decode('utf-8'))
        self.tool_weights = saved['tool_adapter']
        self.adapters = None
        self.device = device

    def install_tools(self):
        if self.adapters is None:
            self.adapters = apply_adapter(self.model, self.tool_weights, self.base_hash, self.vocab_hash)
        return self.adapters

    def generate(self, messages, limit, tools=False):
        message = messages[-1]['content']
        if tools:
            with adapter_enabled(self.install_tools()):
                tool = choose_tool(self.model, self.tokenizer, message)
                if tool:
                    return reply_with_result(self.model, self.tokenizer, message, tool, invoke(tool, message))
        prompt = ''.join(
            f"<|user|> {user_content(turn['content'])} " if turn['role'] == 'user'
            else f"<|ai|> {user_content(turn['content'])} <|eos|>\n"
            for turn in messages
        ) + '<|ai|>'
        tokens = self.tokenizer.encode(prompt).ids
        ids = torch.tensor([tokens], device=self.device)
        stops = {self.tokenizer.token_to_id(tag) for tag in ('<|eos|>', '<|user|>')}
        return self.tokenizer.decode(list(self.model.generate_stream(ids, limit, stop_ids=stops))).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/agm-6-lite.pt')
    parser.add_argument('--prompt')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--tokens', type=int, default=160)
    parser.add_argument('--tools', action=argparse.BooleanOptionalAction, default=False,
                        help='Use /tools on or /tools off to change the mode during chat')
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('--tokens must be positive')

    torch.set_num_threads(2)
    chat = LiteChat(args.model, args.device)
    run_chat(chat.model, chat.tokenizer, args, chat.install_tools)


if __name__ == '__main__':
    main()
