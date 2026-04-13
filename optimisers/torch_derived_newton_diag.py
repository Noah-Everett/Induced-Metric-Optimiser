# optimisers/torch_derived_newton_diag.py
"""
PyTorch implementation of the **derived Newton-targeted diagonal optimizer**.

This optimizer replaces the learnable metric's gradient-ascent s-update
(torch_learnable_diag.py) with a direct EMA towards the Newton target:

    s_{i,t+1} = (1 - beta_s) * s_{i,t} + beta_s * (-log max(H_hat_ii, eps))

where H_hat_ii is the Hutchinson Hessian diagonal estimate.  The derivation
(IMO-83 through IMO-87) shows that this is the unique minimax-optimal diagonal
preconditioner.  No xi, no denominator, no metric_ema (xi=0 => r=1).

Usage (with explicit h_diag)::

    opt = DerivedNewtonDiag(model.parameters(), lr=0.1, ...)
    loss = compute_loss(model, batch)
    loss.backward()
    h_diag = hutchinson_hvp_diag_torch(lambda p: loss_fn(p), list(model.parameters()))
    opt.step(h_diag=h_diag)

Usage (with closure)::

    def closure():
        opt.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        return loss
    opt.step(closure=closure, use_hutchinson=True)

A helper function ``hutchinson_hvp_diag_torch`` is provided for computing h_diag
from a loss function using ``torch.autograd.functional.hvp`` (~2x gradient cost).

References:
- IMO-84: Newton target uniqueness theorem
- IMO-85: Robustness theorem (Hutchinson over secant)
- IMO-86: xi=0 optimality
- IMO-87: Plain embedding non-interference
"""

from __future__ import annotations
from typing import Iterable, List, Optional
import torch
from torch.optim import Optimizer


# ---- Hutchinson helper (PyTorch) -------------------------------------------

def hutchinson_hvp_diag_torch(
    loss_fn,
    params: List[torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Estimate Hessian diagonal via Hutchinson's method with Rademacher vectors.

    Uses ``torch.autograd.functional.hvp`` for the Hessian-vector product.
    Total cost is approximately 2x a single gradient evaluation.

    Parameters
    ----------
    loss_fn : callable
        Scalar loss function: ``loss_fn(*params) -> scalar``.
        Must accept unpacked parameter tensors (not a list).
    params : list of Tensor
        Current parameters (must have requires_grad=True or be inputs to loss_fn).
    generator : torch.Generator, optional
        RNG for Rademacher vector sampling.

    Returns
    -------
    h_diag : list of Tensor
        Estimated Hessian diagonal (same shapes as params).
    """
    v = [
        2.0 * torch.bernoulli(
            torch.full_like(p, 0.5), generator=generator
        ) - 1.0
        for p in params
    ]
    _, hvp = torch.autograd.functional.hvp(loss_fn, tuple(params), tuple(v))
    return [vi * hi for vi, hi in zip(v, hvp)]


# ---- optimizer --------------------------------------------------------------

class DerivedNewtonDiag(Optimizer):
    """Derived Newton-targeted diagonal optimizer (PyTorch).

    The inverse metric diag(exp(s)) is driven towards the Newton target
    s_i* = -log H_ii via an EMA of Hutchinson Hessian diagonal estimates.
    No xi, no denominator, no metric parameterisation variants.

    The ``step()`` method takes ``h_diag`` (list of tensors matching parameter
    shapes) as a keyword argument::

        opt.step(h_diag=[...])

    Use :func:`hutchinson_hvp_diag_torch` to compute ``h_diag``.

    Args:
        params: Iterable of parameters to optimize.
        lr: Parameter step size eta.
        momentum: Momentum coefficient beta_m.
        beta_s: EMA rate for s towards Newton target.
        weight_decay: Decoupled weight decay.
        metric_clip: Symmetric clip bound for s after mean-centering.
        hess_eps: Floor for max(h_diag, eps) before taking log.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.9,
        beta_s: float = 0.1,
        weight_decay: float = 0.0,
        metric_clip: float = 4.0,
        hess_eps: float = 1e-6,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta_s=beta_s,
            weight_decay=weight_decay,
            metric_clip=metric_clip,
            hess_eps=hess_eps,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            grad_params = [p for p in group['params'] if p.requires_grad]
            if not grad_params:
                continue
            p0 = grad_params[0]
            self.state[p0].setdefault('step', 0)
            for p in grad_params:
                st = self.state[p]
                st.setdefault('momentum_buffer', torch.zeros_like(p.data))
                st.setdefault('log_diag', torch.zeros_like(p.data))

    @torch.no_grad()
    def step(
        self,
        closure: Optional[callable] = None,
        h_diag: Optional[List[torch.Tensor]] = None,
    ):
        """Perform a single optimization step.

        Either pass ``h_diag`` (a list of tensors, one per parameter with
        ``requires_grad=True``, in iteration order) or provide a ``closure``
        that recomputes the loss (not yet auto-Hutchinson; pass h_diag).

        Args:
            closure: Optional closure for recomputing the loss.
            h_diag: Hutchinson Hessian diagonal estimates, one tensor per
                parameter (matching the order of param_groups).
        """
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()

        # Build an index into h_diag
        h_idx = 0

        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            beta_s = group['beta_s']
            weight_decay = group['weight_decay']
            clip = abs(group['metric_clip'])
            hess_eps = group['hess_eps']
            lo, hi = -clip, clip

            grad_params = [p for p in group['params'] if p.requires_grad]
            if not grad_params:
                continue

            p0 = grad_params[0]
            gstate = self.state[p0]
            gstate['step'] += 1
            step = gstate['step']

            # --- Metric EMA towards Newton target ---
            for p in grad_params:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = st['log_diag']
                if h_diag is not None:
                    h = h_diag[h_idx]
                    h_idx += 1
                    target = -torch.log(torch.clamp(h, min=hess_eps))
                    s.mul_(1.0 - beta_s).add_(beta_s * target)
                # mean-centre per tensor
                s.add_(-s.mean())
                # clip
                s.clamp_(min=lo, max=hi)

            # --- Momentum with bias correction ---
            for p in grad_params:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st['momentum_buffer']
                s = st['log_diag']

                buf.mul_(mom).add_((1.0 - mom) * p.grad)
                m_corr = buf / (1.0 - mom ** step)

                # Preconditioned step (xi=0 => r=1)
                m_tilde = torch.exp(s) * m_corr

                p.add_(-lr * m_tilde)
                if weight_decay != 0.0:
                    p.add_(-lr * weight_decay * p)

        return loss_val
