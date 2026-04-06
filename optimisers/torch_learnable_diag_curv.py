# optimisers/torch_learnable_diag_curv.py
"""
PyTorch implementation of induced-metric SGD with a curvature-aware learnable diagonal
inverse metric.

Extends the diagonal learnable variant (torch_learnable_diag.py) with a secant-based
Hessian diagonal estimator that drives anti-correlation between metric scale and curvature
sign. Setting curv_beta=0 recovers the original diagonal learnable rule exactly.

- SGDLearnableDiagCurv: f(L)=L (Alg. 1) when log_loss=False
                         f(L)=log L (Alg. 2) when log_loss=True

The curvature-aware metric update is:
  s_i <- s_i + mu * [xi * exp(s_i) * g_i^2
                      - curv_beta * tanh(H_hat_ii / curv_tau) * exp(s_i) * g_i^2]
                - mu * lambda_s * s_i

where H_hat_ii is estimated via secant differences:
  H_hat_ii ≈ (g_i^{(t)} - g_i^{(t-1)}) / (theta_i^{(t)} - theta_i^{(t-1)})

Reference: Phiacta entry "Optimal Curvature Correction and Proposed Update Rule"
           (1998d606-92de-4a33-9345-bcf7809f9a6c)
"""

from __future__ import annotations
from typing import Iterable, Optional
import math
import torch
from torch.optim import Optimizer


class SGDLearnableDiagCurv(Optimizer):
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
        curv_beta: float = 0.05,
        curv_tau: float = 1.0,
        secant_eps: float = 1e-5,
    ):
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
            curv_beta=curv_beta,
            curv_tau=curv_tau,
            secant_eps=secant_eps,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            params_list = [p for p in group['params'] if p.requires_grad]
            if not params_list:
                continue
            p0 = params_list[0]
            state = self.state[p0]
            state.setdefault('step', 0)
            state.setdefault('metric_ema', torch.zeros((), device=p0.device))
            for p in params_list:
                st = self.state[p]
                st.setdefault('momentum_buffer', torch.zeros_like(p.data))
                st.setdefault('log_diag', torch.zeros_like(p.data))
                st.setdefault('prev_grad', torch.zeros_like(p.data))
                st.setdefault('prev_param', p.data.clone())

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None, loss: Optional[float] = None):
        """Performs a single optimization step."""
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
        if loss is None and loss_val is not None:
            loss = float(loss_val)

        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            xi = group['xi']
            beta = group['beta']
            weight_decay = group['weight_decay']
            metric_lr = group['metric_lr']
            metric_reg = group['metric_reg']
            metric_clip = abs(group['metric_clip'])
            log_loss_mode = group['log_loss']
            curv_beta = group['curv_beta']
            curv_tau = group['curv_tau']
            secant_eps = group['secant_eps']

            params_list = [p for p in group['params'] if p.requires_grad]
            if not params_list:
                continue
            p0 = params_list[0]
            gstate = self.state[p0]
            gstate['step'] += 1
            step = gstate['step']
            metric_ema = gstate['metric_ema']

            # Compute v = xi * sum g * (gamma^{-1} g)
            v_accum = 0.0
            for p in params_list:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                g = p.grad
                g_tilde = torch.exp(s) * g
                v_accum = v_accum + torch.sum(g * g_tilde)

            v = xi * v_accum
            metric_ema.mul_(beta).add_((1.0 - beta) * v)
            gstate['metric_ema'] = metric_ema
            v_hat = metric_ema / (1.0 - (beta ** step))

            # r factor
            if log_loss_mode:
                if loss is None:
                    raise RuntimeError("log_loss=True but no `loss` was provided to step().")
                loss_t = loss if torch.is_tensor(loss) else torch.tensor(loss, device=p0.device)
                r = loss_t / (loss_t * loss_t + torch.abs(v_hat))
            else:
                r = 1.0 / (1.0 + torch.abs(v_hat))

            # Update per-parameter momentum, apply inverse metric, and step
            for p in params_list:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                buf = st['momentum_buffer']

                buf.mul_(mom).add_((1.0 - mom) * p.grad)
                m_corr = buf / (1.0 - (mom ** step))
                m_tilde = torch.exp(s) * m_corr

                p.add_(-lr * r * m_tilde)
                if weight_decay != 0.0:
                    p.add_(-lr * weight_decay * p)

            # --- Online metric learning step (curvature-aware) ---
            have_prev = step > 1

            for p in params_list:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                g = p.grad
                g2 = g * g
                exp_s = torch.exp(s)

                # Base adaptive term
                grad_s = xi * exp_s * g2

                # Curvature correction (skip step 1 — no valid secant)
                if have_prev and curv_beta > 0.0:
                    prev_g = st['prev_grad']
                    prev_p = st['prev_param']
                    delta_theta = p.data - prev_p
                    delta_grad = g - prev_g

                    # Guard against small denominators
                    safe = delta_theta.abs() > secant_eps
                    denom = torch.where(safe, delta_theta, torch.ones_like(delta_theta))
                    h_hat = torch.where(safe, delta_grad / denom, torch.zeros_like(g))

                    # Smoothed sign
                    curv_sign = torch.tanh(h_hat / curv_tau)

                    # Subtract correction: shrink s where H>0, grow where H<0
                    grad_s = grad_s - curv_beta * curv_sign * exp_s * g2

                s.add_(metric_lr * grad_s - metric_lr * metric_reg * s)
                # Mean-centre per tensor
                s.add_(-s.mean())
                s.clamp_(min=-metric_clip, max=metric_clip)

                # Store current grad and param for next step's secant
                st['prev_grad'].copy_(g)
                st['prev_param'].copy_(p.data)

        return loss_val
