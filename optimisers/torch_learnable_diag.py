# optimisers/torch_learnable.py
"""
PyTorch implementation of induced-metric SGD with a learnable diagonal inverse metric.

- SGDLearnableDiag: f(L)=L (Alg. 1) when log_loss=False
                     f(L)=log L (Alg. 2) when log_loss=True (requires passing loss to step())

The inverse metric is parameterised as gamma^{-1}_phi = diag(sigma(s)), where sigma
is controlled by ``metric_param`` (see jax_learnable_diag.py for details).

Usage:
    opt = SGDLearnableDiag(model.parameters(), lr=..., momentum=..., xi=..., beta=..., ...)
    # plain:
    loss.backward()
    opt.step()
    # log-loss:
    loss.backward()
    opt.step(loss=loss.item())  # or pass loss tensor

This class performs decoupled weight decay (AdamW-style).
"""

from __future__ import annotations
from typing import Iterable, Optional
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

_VALID_METRIC_PARAMS = frozenset({
    "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
})

# softplus(s) = 1  =>  s = log(e - 1)
_SOFTPLUS_IDENTITY_INIT = float(torch.log(torch.exp(torch.ones(())) - 1.0))


def _torch_sigma(metric_param, s):
    """Apply metric parameterization: exp(s) or softplus(s)."""
    if metric_param == "softplus":
        return F.softplus(s)
    return torch.exp(s)


class SGDLearnableDiag(Optimizer):
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
            lr=lr,
            momentum=momentum,
            xi=xi,
            beta=beta,
            weight_decay=weight_decay,
            metric_lr=metric_lr,
            metric_reg=metric_reg,
            metric_clip=metric_clip,
            log_loss=log_loss,
            metric_param=metric_param,
            max_condition_number=max_condition_number if max_condition_number is not None else 1000.0,
        )
        super().__init__(params, defaults)

        use_softplus = metric_param == "softplus"

        # We store a scalar EMA and a step counter on the first param of each group.
        for group in self.param_groups:
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            state = self.state[p0]
            state.setdefault('step', 0)
            state.setdefault('metric_ema', torch.zeros((), device=p0.device))
            # Per-parameter state: momentum buffer and log_diag (s)
            for p in params:
                st = self.state[p]
                st.setdefault('momentum_buffer', torch.zeros_like(p.data))
                if use_softplus:
                    st.setdefault('log_diag', torch.full_like(p.data, _SOFTPLUS_IDENTITY_INIT))
                else:
                    st.setdefault('log_diag', torch.zeros_like(p.data))

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None, loss: Optional[float] = None):
        """Performs a single optimization step.
        If log_loss=True in the param group, you should pass `loss` (float or 0D tensor)."""
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
            max_cond = group['max_condition_number']
            use_adaptive_clip = mp == "exp_adaptive_clip"

            # Anchor state for scalar EMA + step
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue
            p0 = params[0]
            gstate = self.state[p0]
            gstate['step'] += 1
            step = gstate['step']
            metric_ema = gstate['metric_ema']

            # Compute v = xi * sum g * (sigma(s) * g)
            v_accum = 0.0
            for p in params:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                g = p.grad
                g_tilde = _torch_sigma(mp, s) * g
                v_accum = v_accum + torch.sum(g * g_tilde)

            v = xi * v_accum
            metric_ema.mul_(beta).add_( (1.0 - beta) * v )
            gstate['metric_ema'] = metric_ema
            # Bias correction
            v_hat = metric_ema / (1.0 - (beta ** step))

            # r factor
            if log_loss_mode:
                if loss is None:
                    raise RuntimeError("log_loss=True but no `loss` was provided to step().")
                # allow tensor or float
                loss_t = loss if torch.is_tensor(loss) else torch.tensor(loss, device=p0.device)
                r = loss_t / (loss_t * loss_t + torch.abs(v_hat))
            else:
                r = 1.0 / (1.0 + torch.abs(v_hat))

            # Update per-parameter momentum, apply inverse metric, and step
            for p in params:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                buf = st['momentum_buffer']

                # Momentum update
                buf.mul_(momentum).add_( (1.0 - momentum) * p.grad )
                # Bias-corrected momentum
                m_corr = buf / (1.0 - (momentum ** step))
                # Apply inverse metric
                m_tilde = _torch_sigma(mp, s) * m_corr

                # Parameter update (decoupled weight decay)
                if weight_decay != 0.0:
                    p_orig = p.data.clone()
                    p.add_(-lr * r * m_tilde)
                    p.add_(-lr * weight_decay * p_orig)
                else:
                    p.add_(-lr * r * m_tilde)

            # --- Online metric learning step for s ---
            for p in params:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                g2 = p.grad * p.grad

                if mp == "softplus":
                    grad_s = xi * F.softplus(s) * torch.sigmoid(s) * g2
                    s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)
                elif mp == "exp_matched_reg":
                    grad_s = xi * torch.exp(s) * g2
                    s.add_(metric_lr * grad_s - metric_lr * metric_reg * torch.exp(s))
                elif mp == "exp_norm_grad":
                    grad_s = xi * g2
                    s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)
                else:
                    # "exp" and "exp_adaptive_clip"
                    grad_s = xi * torch.exp(s) * g2
                    s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)

                # mean-centre per tensor
                s.add_( -s.mean() )

            # Clip: adaptive or fixed
            if use_adaptive_clip:
                all_s = torch.cat([self.state[p]['log_diag'].ravel()
                                   for p in params if p.grad is not None])
                s_median = all_s.median()
                half_log_cond = 0.5 * math.log(max_cond)
                lo = s_median - half_log_cond
                hi = s_median + half_log_cond
                for p in params:
                    if p.grad is None:
                        continue
                    self.state[p]['log_diag'].clamp_(min=float(lo), max=float(hi))
            else:
                for p in params:
                    if p.grad is None:
                        continue
                    self.state[p]['log_diag'].clamp_(min=-metric_clip, max=metric_clip)

        return loss_val
