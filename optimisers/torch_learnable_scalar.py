# optimisers/torch_scalar_learnable.py
"""
PyTorch optimiser: induced-metric SGD with a **learnable scalar inverse metric**.

- SGDLearnableScalar(..., log_loss=False)  # Alg. 1
- SGDLearnableScalar(..., log_loss=True)   # Alg. 2 (pass `loss` to step())

gamma^{-1}_phi = sigma(s) * I, where sigma is controlled by ``metric_param``.
We learn scalar s online by ascending v = xi * sigma(s) * sum(g^2), plus
regularization on s, and clipping in log-domain for stability. Momentum and
decoupled weight decay match the repo.
"""

from __future__ import annotations
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

_VALID_METRIC_PARAMS = frozenset({
    "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
})

_SOFTPLUS_IDENTITY_INIT = float(torch.log(torch.exp(torch.ones(())) - 1.0))


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
        metric_param: str = "exp",
        max_condition_number: Optional[float] = None,
    ):
        if metric_param not in _VALID_METRIC_PARAMS:
            raise ValueError(f"Unknown metric_param={metric_param!r}. "
                             f"Valid: {sorted(_VALID_METRIC_PARAMS)}")
        defaults = dict(
            lr=lr, momentum=momentum, xi=xi, beta=beta, weight_decay=weight_decay,
            metric_lr=metric_lr, metric_reg=metric_reg, metric_clip=metric_clip,
            log_loss=log_loss, metric_param=metric_param,
        )
        super().__init__(params, defaults)

        use_softplus = metric_param == "softplus"

        # Each param group keeps: step, metric_ema, and a scalar log_scale.
        for group in self.param_groups:
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            gstate = self.state[p0]
            gstate.setdefault('step', 0)
            gstate.setdefault('metric_ema', torch.zeros((), device=p0.device))
            init_val = _SOFTPLUS_IDENTITY_INIT if use_softplus else 0.0
            gstate.setdefault('log_scale', torch.tensor(init_val, device=p0.device))

            for p in params:
                st = self.state[p]
                st.setdefault('momentum_buffer', torch.zeros_like(p.data))

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None, loss: Optional[float] = None):
        """Performs a single optimization step. If log_loss=True, pass `loss`."""
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
            mp = group['metric_param']

            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            gstate = self.state[p0]
            gstate['step'] += 1
            step = gstate['step']
            metric_ema = gstate['metric_ema']
            s = gstate['log_scale']

            if mp == "softplus":
                alpha = F.softplus(s)
            else:
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

            # online metric update for scalar s
            if mp == "softplus":
                grad_s = xi * F.softplus(s) * torch.sigmoid(s) * v_accum
                s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)
            elif mp == "exp_matched_reg":
                grad_s = xi * alpha * v_accum
                s.add_(metric_lr * grad_s - metric_lr * metric_reg * alpha)
            elif mp == "exp_norm_grad":
                grad_s = xi * v_accum
                s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)
            else:
                # "exp" and "exp_adaptive_clip"
                grad_s = xi * alpha * v_accum
                s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)

            # Scalar: adaptive clip is no-op (condition number = 1), use fixed clip
            s.clamp_(min=-metric_clip, max=metric_clip)
            gstate['log_scale'] = s

        return loss_val
