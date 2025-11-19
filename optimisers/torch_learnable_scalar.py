# optimisers/torch_scalar_learnable.py
"""
PyTorch optimiser: induced-metric SGD with a **learnable scalar inverse metric**.

- SGDLearnableScalar(..., log_loss=False)  # Alg. 1
- SGDLearnableScalar(..., log_loss=True)   # Alg. 2 (pass `loss` to step())

gamma^{-1}_phi = exp(s) * I, with group-level scalar s updated online by ascending
v = xi * sum(g * (gamma^{-1} g)) = xi * exp(s) * sum(g^2), plus L2 regularization on s,
and clipping in log-domain for stability. Momentum and decoupled weight decay match the repo.
"""

from __future__ import annotations
from typing import Iterable, Optional
import torch
from torch.optim import Optimizer

class SGDLearnableScalar(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.9,
        xi: float = 0.1,
        beta: float = 0.8,
        weight_decay: float = 0.0,
        metric_lr: float = 1e-3,
        metric_reg: float = 1e-4,
        metric_clip: float = 4.0,
        log_loss: bool = False,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, xi=xi, beta=beta, weight_decay=weight_decay,
            metric_lr=metric_lr, metric_reg=metric_reg, metric_clip=metric_clip,
            log_loss=log_loss,
        )
        super().__init__(params, defaults)

        # Each param group keeps: step, metric_ema, and a scalar log_scale on the device of the first param.
        for group in self.param_groups:
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            gstate = self.state[p0]
            gstate.setdefault('step', 0)
            gstate.setdefault('metric_ema', torch.zeros((), device=p0.device))
            gstate.setdefault('log_scale', torch.zeros((), device=p0.device))

            for p in params:
                st = self.state[p]
                st.setdefault('momentum_buffer', torch.zeros_like(p.data))

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None, loss: Optional[float] = None):
        """Performs a single optimization step. If log_loss=True, pass `loss` (float or 0D tensor)."""
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
        if loss is None and loss_val is not None:
            loss = float(loss_val)

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            xi = group['xi']
            beta = group['beta']
            weight_decay = group['weight_decay']
            metric_lr = group['metric_lr']
            metric_reg = group['metric_reg']
            metric_clip = abs(group['metric_clip'])
            log_loss_mode = group['log_loss']

            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            gstate = self.state[p0]
            gstate['step'] += 1
            step = gstate['step']
            metric_ema = gstate['metric_ema']
            s = gstate['log_scale']
            alpha = torch.exp(s)

            # compute v = xi * alpha * sum g^2 (across group)
            v_accum = 0.0
            for p in params:
                if p.grad is None:
                    continue
                v_accum = v_accum + torch.sum(p.grad * p.grad)
            v = xi * alpha * v_accum

            # EMA of v and bias correction
            metric_ema.mul_(beta).add_( (1.0 - beta) * v )
            v_hat = metric_ema / (1.0 - (beta ** step))
            gstate['metric_ema'] = metric_ema

            # r factor
            if log_loss_mode:
                if loss is None:
                    raise RuntimeError("log_loss=True but no `loss` passed to step().")
                loss_t = loss if torch.is_tensor(loss) else torch.tensor(loss, device=p0.device)
                r = loss_t / (loss_t * loss_t + torch.abs(v_hat))
            else:
                r = 1.0 / (1.0 + torch.abs(v_hat))

            # momentum update and parameter step (decoupled WD)
            for p in params:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st['momentum_buffer']
                buf.mul_(momentum).add_( (1.0 - momentum) * p.grad )
                m_corr = buf / (1.0 - (momentum ** step))

                m_tilde = alpha * m_corr
                p.add_( -lr * r * m_tilde )
                if weight_decay != 0.0:
                    p.add_( -lr * weight_decay * p )

            # online metric update for scalar s: d/ds v = xi * exp(s) * sum g^2
            grad_s = xi * alpha * v_accum
            s.add_( metric_lr * grad_s - metric_lr * metric_reg * s )
            s.clamp_(min=-metric_clip, max=metric_clip)
            gstate['log_scale'] = s

        return loss_val
