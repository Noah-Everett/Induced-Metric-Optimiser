"""
PyTorch implementation of SGD with an induced metric that has non-zero
off-diagonal blocks in the ambient metric, matching the JAX version in
jax_offdiag.py.

Pullback metric:
    g_pb = gamma * I + xi (a l^T + l b^T + l l^T),

with configurable choices for:
    - l (base "gradient-like" direction),
    - a, b (additional directions),
    - update vector r.

The optimizer applies:
    delta_theta = -lr * g_pb^{-1} r

via a rank-2 Woodbury update, requiring O(d) work and a 2×2 solve.

Class:
    CustomSGDOffDiag

Features:
    * mode-based choices for l, a, b:
        - base_mode: "grad", "momentum"
        - a_mode: "zero", "grad", "momentum", "params"
        - b_mode: "same_as_a", "zero", "grad", "momentum", "params"

    * static a and b:
        - a_static: either a 1D tensor (flattened direction) or a sequence
          of tensors matching params (will be flattened)
        - b_static: same options

    * dynamic a and b via callbacks:
        - a_fn(params_with_grad, grads, m_hat_list, step) -> sequence of
          tensors or a 1D tensor
        - b_fn(params_with_grad, grads, m_hat_list, step) -> sequence/1D

    Precedence:
        a: a_fn -> a_static -> a_mode
        b: b_fn -> b_static -> b_mode / "same_as_a"

Note:
    - Static / callback-based 1D tensors must have the same total length
      as the flattened parameter vector in the corresponding param group.
"""

from typing import Optional, Callable, Iterable, List

import torch
from torch.optim.optimizer import Optimizer


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into a single 1D tensor."""
    tensors = list(tensors)
    if len(tensors) == 0:
        return torch.tensor([], dtype=torch.float32)
    flats = [t.reshape(-1) for t in tensors]
    return torch.cat(flats, dim=0)


def _apply_inverse_metric_offdiag_torch(
    r_flat: torch.Tensor,
    l_flat: torch.Tensor,
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    gamma: float,
    xi: float,
) -> torch.Tensor:
    """Torch version of (gamma I + xi (a l^T + l b^T + l l^T))^{-1} r."""
    if r_flat.numel() == 0:
        return r_flat

    device = r_flat.device
    dtype = r_flat.dtype

    inv_gamma = 1.0 / gamma

    # U and V: (d, 2)
    U = torch.stack([a_flat, l_flat], dim=1)
    V = torch.stack([l_flat, b_flat + l_flat], dim=1)

    alpha = xi * inv_gamma  # xi/gamma
    B = alpha * U           # (d, 2)
    z = inv_gamma * r_flat  # (d,)

    # M = I_2 + V^T B  (2x2)
    M = torch.eye(2, dtype=dtype, device=device) + V.t().mm(B)  # (2, 2)

    # rhs = V^T z  (2,)
    rhs = V.t().mv(z)

    # Solve 2x2
    w = torch.linalg.solve(M, rhs)

    # y = z - B @ w
    y_flat = z - B.mv(w)
    return y_flat


class CustomSGDOffDiag(Optimizer):
    """PyTorch SGD with a non-zero off-diagonal induced metric.

    This matches the JAX custom_sgd_offdiag() implementation in spirit.

    Args:
        params: Iterable of parameters.
        lr: Learning rate.
        momentum: Momentum factor (0 <= momentum < 1).
        xi: Strength of the rank-2 correction in the metric.
        weight_decay: L2 weight decay coefficient (Euclidean).
        gamma: Scalar base metric factor (gamma * I).
        base_mode: Which vector to use as l in the metric:
            - "grad":      l = gradients
            - "momentum":  l = bias-corrected momentum
        a_mode: How to choose a if neither a_fn nor a_static is provided:
            - "zero":      a = 0
            - "grad":      a = gradients
            - "momentum":  a = bias-corrected momentum
            - "params":    a = parameters
        b_mode: How to choose b if neither b_fn nor b_static is provided:
            - "same_as_a": b = a
            - "zero":      b = 0
            - "grad":      b = gradients
            - "momentum":  b = bias-corrected momentum
            - "params":    b = parameters
        use_momentum_for_update: If True, precondition the bias-corrected
            momentum; else precondition l directly.
        a_static: Optional static direction for a. Either:
            - a 1D tensor of length equal to total params in the group, or
            - a sequence of tensors matching the params; will be flattened.
        b_static: Same as a_static, but for b.
        a_fn: Optional callable:
            a_fn(params_with_grad, grads, m_hat_list, step) -> sequence of
                tensors or a 1D tensor.
        b_fn: Optional callable with the same signature, for b.

    Precedence:
        a: a_fn -> a_static -> a_mode
        b: b_fn -> b_static -> b_mode / "same_as_a"
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        xi: float = 0.1,
        weight_decay: float = 0.0,
        gamma: float = 1.0,
        base_mode: str = "grad",
        a_mode: str = "momentum",
        b_mode: str = "same_as_a",
        use_momentum_for_update: bool = True,
        a_static: Optional[object] = None,
        b_static: Optional[object] = None,
        a_fn: Optional[Callable[[List[torch.nn.Parameter], List[torch.Tensor], List[torch.Tensor], int], object]] = None,
        b_fn: Optional[Callable[[List[torch.nn.Parameter], List[torch.Tensor], List[torch.Tensor], int], object]] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if xi < 0.0:
            raise ValueError(f"Invalid xi value: {xi}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            xi=xi,
            weight_decay=weight_decay,
            gamma=gamma,
            base_mode=base_mode,
            a_mode=a_mode,
            b_mode=b_mode,
            use_momentum_for_update=use_momentum_for_update,
            a_static=a_static,
            b_static=b_static,
            a_fn=a_fn,
            b_fn=b_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global step (for momentum bias correction)
        if "global" not in self.state:
            self.state["global"] = dict(step=0)
        global_state = self.state["global"]
        global_state["step"] += 1
        step = global_state["step"]

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            xi = group["xi"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            base_mode = group["base_mode"]
            a_mode = group["a_mode"]
            b_mode = group["b_mode"]
            use_momentum_for_update = group["use_momentum_for_update"]
            a_static = group["a_static"]
            b_static = group["b_static"]
            a_fn = group["a_fn"]
            b_fn = group["b_fn"]

            params_with_grad: List[torch.nn.Parameter] = []
            grads: List[torch.Tensor] = []
            momentums: List[torch.Tensor] = []

            # Update momentum buffers and collect params/grads/momentum
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = dict(
                        momentum_buffer=torch.zeros_like(p.data)
                    )

                state_p = self.state[p]
                buf = state_p["momentum_buffer"]
                grad = p.grad.data

                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                state_p["momentum_buffer"] = buf

                params_with_grad.append(p)
                grads.append(grad)
                momentums.append(buf)

            if len(params_with_grad) == 0:
                continue

            device = params_with_grad[0].device

            # Bias-corrected momentum
            bias_correction = 1.0 - momentum ** step
            m_hat_list = [m / bias_correction for m in momentums]

            # Choose l
            if base_mode == "grad":
                l_list = grads
            elif base_mode == "momentum":
                l_list = m_hat_list
            else:
                raise ValueError(f"Unknown base_mode: {base_mode}")

            # Choose r (vector to be preconditioned)
            r_list = m_hat_list if use_momentum_for_update else l_list

            # ----- helper to flatten static or fn outputs -----
            def _to_flat_from_spec(spec, fallback_list: List[torch.Tensor]) -> torch.Tensor:
                """Convert 'spec' to flat tensor.

                spec can be:
                    - None
                    - 1D Tensor
                    - sequence of Tensors (same shapes as fallback_list)
                """
                if spec is None:
                    # Should not be called in that case
                    raise RuntimeError("spec is None in _to_flat_from_spec")

                if torch.is_tensor(spec):
                    return spec.to(device=device, dtype=fallback_list[0].dtype)

                # Assume spec is an iterable of tensors
                return _flatten_tensors(spec)

            # ----- build a_flat -----
            if a_fn is not None:
                a_spec = a_fn(params_with_grad, grads, m_hat_list, step)
                a_flat = _to_flat_from_spec(a_spec, grads)
            elif a_static is not None:
                a_flat = _to_flat_from_spec(a_static, grads)
            else:
                # Mode-based a_list
                if a_mode == "zero":
                    a_list = [torch.zeros_like(g) for g in grads]
                elif a_mode == "grad":
                    a_list = grads
                elif a_mode == "momentum":
                    a_list = m_hat_list
                elif a_mode == "params":
                    a_list = [p.data for p in params_with_grad]
                else:
                    raise ValueError(f"Unknown a_mode: {a_mode}")
                a_flat = _flatten_tensors(a_list)

            # ----- build l_flat and r_flat -----
            l_flat = _flatten_tensors(l_list)
            r_flat = _flatten_tensors(r_list)

            # Sanity check for a_flat length
            if a_flat.numel() != r_flat.numel():
                raise ValueError(
                    f"a_flat has {a_flat.numel()} elements, but r_flat has "
                    f"{r_flat.numel()} elements. They must match."
                )

            # ----- build b_flat -----
            if b_fn is not None:
                b_spec = b_fn(params_with_grad, grads, m_hat_list, step)
                b_flat = _to_flat_from_spec(b_spec, grads)
            elif b_static is not None:
                b_flat = _to_flat_from_spec(b_static, grads)
            else:
                if b_mode == "same_as_a":
                    b_flat = a_flat
                else:
                    # Mode-based b_list
                    if b_mode == "zero":
                        b_list = [torch.zeros_like(g) for g in grads]
                    elif b_mode == "grad":
                        b_list = grads
                    elif b_mode == "momentum":
                        b_list = m_hat_list
                    elif b_mode == "params":
                        b_list = [p.data for p in params_with_grad]
                    else:
                        raise ValueError(f"Unknown b_mode: {b_mode}")
                    b_flat = _flatten_tensors(b_list)

            if b_flat.numel() != r_flat.numel():
                raise ValueError(
                    f"b_flat has {b_flat.numel()} elements, but r_flat has "
                    f"{r_flat.numel()} elements. They must match."
                )

            # Apply inverse metric in flat space
            y_flat = _apply_inverse_metric_offdiag_torch(
                r_flat=r_flat,
                l_flat=l_flat,
                a_flat=a_flat,
                b_flat=b_flat,
                gamma=gamma,
                xi=xi,
            )

            # Scatter y_flat back to parameters
            offset = 0
            for p in params_with_grad:
                numel = p.numel()
                y_slice = y_flat[offset:offset + numel].view_as(p.data)
                offset += numel

                # Euclidean weight decay
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Metric-preconditioned update
                p.data.add_(y_slice, alpha=-lr)

        return loss
