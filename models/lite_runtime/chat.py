"""Terminal chat using the same frozen tokenizer and prompt format as the UI."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from checkpoint import load_model, tokenizer_hash


def main(argv=None, root=None, description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--checkpoint', default='out/agmgpt_6_it.pt')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--prompt')
    p.add_argument('--tokens', type=int, default=160)
    p.add_argument('--tools', action=argparse.BooleanOptionalAction, default=False,
                   help='Enable or disable local tool use')
    p.add_argument('--tool-adapter', type=Path, default=Path('out/tool-use-lora.pt'))
    args = p.parse_args(argv)
    if args.tokens < 1:
        p.error('--tokens must be positive')
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    torch.set_num_threads(2)
    base = root / args.checkpoint
    tokfile = root / 'data/tokenizer.json'
    model = load_model(base, args.device, tokenizer_hash(tokfile))
    tok = Tokenizer.from_file(str(tokfile))
    from tool_lora import load_adapter

    def install_tools():
        return load_adapter(model, root / args.tool_adapter, base, tokfile)

    run_chat(model, tok, args, install_tools)


def run_chat(model, tok, args, install_tools):
    from chat_tools import choose_tool, invoke, reply_with_result
    from tool_lora import adapter_enabled
    adapters = None
    tools_enabled = args.tools
    history = ''
    while True:
        try:
            message = args.prompt if args.prompt is not None else input('You: ')
        except (EOFError, KeyboardInterrupt):
            break
        command = message.strip().lower()
        if command in ('/quit', '/exit'):
            break
        if args.prompt is None and command in ('/tools on', '/tools off'):
            tools_enabled = command == '/tools on'
            print(f'Tools {"on" if tools_enabled else "off"}.')
            continue
        prompt = history + f'<|user|> {message} <|ai|>'
        if tools_enabled:
            if adapters is None:
                adapters = install_tools()
            with adapter_enabled(adapters):
                tool = choose_tool(model, tok, message)
                if tool:
                    result = invoke(tool, message)
                    reply = reply_with_result(model, tok, message, tool, result)
                    print('AGM:', reply)
                    history = prompt + ' ' + reply + ' <|eos|>\n'
            if tool:
                if args.prompt is not None:
                    break
                continue
        x = torch.tensor([tok.encode(prompt).ids], device=args.device)
        ids, previous = [], ''
        print('AGM: ', end='', flush=True)
        for token in model.generate_stream(x, args.tokens, stop_ids={tok.token_to_id('<|eos|>'), tok.token_to_id('<|user|>')}):
            ids.append(token)
            reply = tok.decode(ids)
            if not reply.endswith('\ufffd'):
                print(reply[len(previous):], end='', flush=True)
                previous = reply
        print()
        history = prompt + ' ' + tok.decode(ids) + ' <|eos|>\n'
        if args.prompt is not None:
            break


if __name__ == '__main__':
    main()
