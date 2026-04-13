# optimisers/test_torch_newton_diag.py
"""
PyTorch-specific functional tests for the derived Newton-targeted diagonal optimizer.

Tests:
1. hutchinson_hvp_diag_torch accuracy on diagonal quadratic
2. beta_s=0 freezes metric at initialisation
3. Saddle differentiation (positive vs negative curvature)
4. h_diag=None fallback (metric unchanged, momentum still works)
5. Mixed requires_grad param groups

Run: PYTHONPATH=repo .conda/bin/python repo/optimisers/test_torch_newton_diag.py
"""

import torch
from optimisers.torch_derived_newton_diag import (
    DerivedNewtonDiag,
    hutchinson_hvp_diag_torch,
)


# ---- Test 1: Hutchinson estimator accuracy ----

def test_hutchinson_accuracy():
    """Averaged Hutchinson samples should match true Hessian diagonal."""
    a, b = 3.0, 7.0

    # loss_fn must accept unpacked params (torch.autograd.functional.hvp unpacks)
    def loss_fn(x, y):
        return a * x**2 + b * y**2

    params = [torch.tensor(1.0), torch.tensor(1.0)]
    true_diag = [2.0 * a, 2.0 * b]

    n_samples = 500
    gen = torch.Generator().manual_seed(42)
    h_sum = [torch.zeros_like(p) for p in params]
    for _ in range(n_samples):
        h = hutchinson_hvp_diag_torch(loss_fn, params, generator=gen)
        for i in range(len(h_sum)):
            h_sum[i] += h[i]
    h_mean = [s / n_samples for s in h_sum]

    err = max(abs(float(h_mean[i]) - true_diag[i]) for i in range(2))
    print(f"[Hutchinson accuracy] mean={[float(h) for h in h_mean]}, "
          f"true={true_diag}, max_err={err:.2e}")
    # For a purely diagonal quadratic, each Rademacher sample gives exact H_ii
    assert err < 1e-5, f"Hutchinson mean too far from truth: err={err}"
    print("PASS: Hutchinson accuracy (PyTorch)")


# ---- Test 2: beta_s=0 freezes metric ----

def test_beta_s_zero():
    """With beta_s=0, s should stay at initialisation (all zeros)."""
    w = torch.randn(4, 3, requires_grad=True)
    opt = DerivedNewtonDiag([w], lr=0.01, momentum=0.9, beta_s=0.0, metric_clip=4.0)

    h_diag = [torch.abs(torch.randn(4, 3)) + 0.1]
    for _ in range(20):
        w.grad = torch.randn_like(w) * 0.1
        opt.step(h_diag=h_diag)

    s = opt.state[w]['log_diag']
    s_max = float(s.abs().max())
    print(f"[beta_s=0] max |s| = {s_max:.2e}")
    assert s_max < 1e-10, f"s should be zero with beta_s=0, got max |s|={s_max}"
    print("PASS: beta_s=0 freezes metric (PyTorch)")


# ---- Test 3: Saddle differentiation ----

def test_saddle_differentiation():
    """On L = x^2 - y^2, H = diag(2, -2).

    Positive curvature (H=2 > eps): s_1 = -log(2) < 0
    Negative curvature (H=-2, clamped to eps): s_2 = -log(eps) >> 0
    So s_1 < s_2 after mean-centering.
    """
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = DerivedNewtonDiag(
        [x], lr=0.01, momentum=0.0, beta_s=0.2,
        weight_decay=0.0, metric_clip=10.0, hess_eps=1e-6,
    )

    h_diag = [torch.tensor([2.0, -2.0])]
    for _ in range(50):
        x.grad = torch.tensor([2.0 * x[0].item(), -2.0 * x[1].item()])
        opt.step(h_diag=h_diag)

    s = opt.state[x]['log_diag']
    s1, s2 = float(s[0]), float(s[1])
    print(f"[Saddle] s1={s1:.4f} (H=2), s2={s2:.4f} (H=-2)")
    assert s1 < s2, f"Expected s1 < s2, got s1={s1:.4f}, s2={s2:.4f}"
    print("PASS: saddle differentiation (PyTorch)")


# ---- Test 4: h_diag=None fallback ----

def test_h_diag_none_fallback():
    """When h_diag=None, metric stays at init (zeros) but momentum still works."""
    w = torch.randn(3, 2, requires_grad=True)
    opt = DerivedNewtonDiag([w], lr=0.01, momentum=0.9, beta_s=0.1)

    w_init = w.data.clone()
    for _ in range(10):
        w.grad = torch.randn_like(w) * 0.1
        opt.step()  # no h_diag

    # s should remain zero (no h_diag to EMA towards)
    s = opt.state[w]['log_diag']
    s_max = float(s.abs().max())
    print(f"[h_diag=None] max |s| = {s_max:.2e}")
    assert s_max < 1e-10, f"s should be zero without h_diag, got {s_max}"

    # But params should have moved (momentum still works with exp(0)=1)
    param_diff = float((w.data - w_init).abs().max())
    print(f"[h_diag=None] param diff = {param_diff:.6f}")
    assert param_diff > 1e-4, f"Params should have moved, got diff={param_diff}"
    print("PASS: h_diag=None fallback (PyTorch)")


# ---- Test 5: Mixed requires_grad param groups ----

def test_mixed_requires_grad():
    """Optimizer handles a mix of requires_grad=True and False params."""
    w1 = torch.randn(4, 3, requires_grad=True)
    w2 = torch.randn(2, 2, requires_grad=False)  # frozen
    w3 = torch.randn(3, requires_grad=True)

    opt = DerivedNewtonDiag([w1, w2, w3], lr=0.01, momentum=0.9, beta_s=0.1)

    # h_diag only for requires_grad params (w1 and w3)
    h_diag = [torch.abs(torch.randn(4, 3)) + 0.1, torch.abs(torch.randn(3)) + 0.1]

    w1_init = w1.data.clone()
    w2_init = w2.data.clone()
    w3_init = w3.data.clone()

    for _ in range(10):
        w1.grad = torch.randn_like(w1) * 0.1
        # w2 has no grad (frozen)
        w3.grad = torch.randn_like(w3) * 0.1
        opt.step(h_diag=h_diag)

    w1_moved = float((w1.data - w1_init).abs().max())
    w2_moved = float((w2.data - w2_init).abs().max())
    w3_moved = float((w3.data - w3_init).abs().max())

    print(f"[Mixed requires_grad] w1 diff={w1_moved:.6f}, "
          f"w2 diff={w2_moved:.6f}, w3 diff={w3_moved:.6f}")
    assert w1_moved > 1e-4, f"w1 should have moved: {w1_moved}"
    assert w2_moved == 0.0, f"w2 (frozen) should not move: {w2_moved}"
    assert w3_moved > 1e-4, f"w3 should have moved: {w3_moved}"
    print("PASS: mixed requires_grad (PyTorch)")


if __name__ == "__main__":
    print("=" * 60)
    print("Derived Newton-targeted optimizer PyTorch tests")
    print("=" * 60)
    print()
    test_hutchinson_accuracy()
    print()
    test_beta_s_zero()
    print()
    test_saddle_differentiation()
    print()
    test_h_diag_none_fallback()
    print()
    test_mixed_requires_grad()
    print()
    print("=" * 60)
    print("ALL PYTORCH TESTS PASSED")
    print("=" * 60)
