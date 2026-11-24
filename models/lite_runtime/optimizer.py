"""Single-device optimizer for the AgmGPT training checkpoint layout."""
import torch

POLAR_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, .5000178451051316),
    (3.8916678022926607, -2.772484153217685, .5060648178503393),
    (3.285753657755655, -2.3681294933425376, .46449024233003106),
    (2.3465413258596377, -1.7097828382687081, .42323551169305323),
)


class TrainingOptimizer(torch.optim.Optimizer):
    def __init__(self, decoder, weight_decay):
        scale = (decoder.config.n_embd / 768) ** -.5
        groups = [
            (list(decoder.lm_head.parameters()), .004 * scale, (.8, .96), .01),
            (list(decoder.transformer.wte.parameters()), .2 * scale, (.8, .995), .001),
            (list(decoder.value_embeds.parameters()), .1 * scale, (.8, .995), .01),
            ([decoder.resid_lambdas], .005, (.8, .95), .05),
            ([decoder.x0_lambdas], .5, (.96, .95), 0.),
            ([decoder.smear_gate.weight, decoder.smear_lambda, decoder.backout_lambda],
             .2, (.8, .95), 0.),
        ]
        param_groups = [
            dict(kind='adamw', params=params, lr=lr, betas=betas, eps=1e-10, weight_decay=decay)
            for params, lr, betas, decay in groups
        ]
        matrices = list(decoder.transformer.h.parameters())
        for shape in sorted({p.shape for p in matrices}):
            param_groups.append(dict(
                kind='muon', params=[p for p in matrices if p.shape == shape],
                lr=.02, momentum=.95, ns_steps=5, beta2=.9, weight_decay=weight_decay,
            ))
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'muon':
                self._muon(group)
            else:
                self._adamw(group)

    def _adamw(self, group):
        beta1, beta2 = group['betas']
        lr = group['lr']
        for param in group['params']:
            if param.grad is None:
                continue
            state = self.state[param]
            if not state:
                state.update(step=0, exp_avg=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))
            state['step'] += 1
            first = state['exp_avg'].float().lerp_(param.grad.float(), 1 - beta1)
            second = state['exp_avg_sq'].float().lerp_(param.grad.float().square(), 1 - beta2)
            step = state['step']
            denom = (second / (1 - beta2 ** step)).sqrt().add_(group['eps'])
            updated = param.float().mul_(1 - lr * group['weight_decay'])
            updated.addcdiv_(first, denom, value=-lr / (1 - beta1 ** step))
            param.copy_(updated)
            state['exp_avg'].copy_(first)
            state['exp_avg_sq'].copy_(second)

    def _muon(self, group):
        params = group['params']
        grads = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        rows, cols = grads.shape[-2:]
        axis = -1 if rows >= cols else -2
        state = self.state[params[0]]
        if not state:
            state['momentum_buffer'] = torch.zeros_like(grads)
            state['second_momentum_buffer'] = torch.zeros_like(grads.sum(dim=axis, keepdim=True))
        momentum = group['momentum']
        first = state['momentum_buffer'].lerp_(grads, 1 - momentum)
        update = grads.lerp_(first, momentum)

        norm = update.norm(dim=(-2, -1), keepdim=True)
        row_norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        update = update * (norm / (rows ** .5 * row_norm))
        update = update / (1.01 * update.norm(dim=(-2, -1), keepdim=True) + 1e-6)
        tall = rows > cols
        if tall:
            update = update.mT
        for a, b, c in POLAR_COEFFICIENTS[:group['ns_steps']]:
            gram = update @ update.mT
            update = a * update + (b * gram + c * (gram @ gram)) @ update
        if tall:
            update = update.mT
        update = update * (min(rows, cols) ** .5 / update.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))

        variance = update.square().mean(dim=axis, keepdim=True)
        second = state['second_momentum_buffer'].lerp_(variance, 1 - group['beta2'])
        scaled = update * second.clamp_min(1e-10).rsqrt()
        scaled *= update.norm(dim=(-2, -1), keepdim=True) / scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)

        weights = torch.stack(params)
        decay = group['weight_decay'] * weights * ((scaled * weights) >= 0)
        weights -= group['lr'] * max(1., rows / cols) ** .5 * (scaled + decay)
        for param, value in zip(params, weights):
            if param.grad is not None:
                param.copy_(value)
