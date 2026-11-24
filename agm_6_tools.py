"""LoRA tool routing for AGM 6."""
import torch
from torch.nn import functional as F

from chat_tools import ROUTES, grounded_reply, invoke, user_content


def route_ids(model, messages):
    return [model.tokenizer.encode(model.format_chat([
        {'role': 'user', 'content': 'Choose local tool for: ' + user_content(message).strip()}
    ]), add_special_tokens=False).ids for message in messages]


def label_ids(model):
    ids = [model.tokenizer.encode(label, add_special_tokens=False).ids for label in 'ABCD']
    if any(len(item) != 1 for item in ids):
        raise ValueError('Tool labels must each use one token')
    return [item[0] for item in ids]


def scores(features, base, down, up):
    return F.linear(features, base) + F.linear(F.linear(features, down), up)


class ToolRouter:
    def __init__(self, model, path):
        self.model = model
        weights = torch.load(path, map_location=model.device, weights_only=True)
        self.base = model.embedding[label_ids(model)].float()
        self.down, self.up = weights['down'], weights['up']

    @torch.inference_mode()
    def reply(self, message):
        ids = torch.tensor(route_ids(self.model, [message]), device=self.model.device)
        features = self.model.forward(ids, feature_positions=[ids.shape[1] - 1]).float()
        probabilities = scores(features, self.base, self.down, self.up).softmax(-1)[0]
        index = int(probabilities.argmax())
        if index and probabilities[index] >= .55:
            tool = ROUTES[index]
            return grounded_reply(tool, invoke(tool, message))
        return None
