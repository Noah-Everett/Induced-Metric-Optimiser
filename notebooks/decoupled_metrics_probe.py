#!/usr/bin/env python3
"""
Decoupled Metrics Probe (C2-decoupled)

Tests the hypothesis that using TWO separate metric parameter vectors resolves
the fundamental tension discovered in C1-num:

  - s^{precond}: controls preconditioning (appears in the parameter update)
  - s^{curv}: controls curvature correction (appears in the Riemannian Hessian)

The tension: for geodesic convexity at a saddle L = x^2 - y^2, we need
s_1 >> s_2 (large metric along positive-curvature x). But the update
delta_theta_i ~ e^{s_i} means small e^{s_2} -> tiny step in escape direction.

Decoupled approach:
  - s^{precond} uses ESCAPE rule: s large where H < 0 (fast saddle escape)
  - s^{curv} uses GEODESIC rule: s large where H > 0 (geodesic convexity)
  - Parameter update uses s^{precond}; Riemannian Hessian diagnostic uses s^{curv}

Run:
  PYTHONPATH=repo .conda/bin/python repo/notebooks/decoupled_metrics_probe.py
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LR = 0.01         # parameter learning rate
MU = 0.1          # metric learning rate
XI = 1.0          # loss-dimension weight
LAMBDA_S = 0.01   # metric regularisation
METRIC_CLIP = 4.0 # bounds on s_i
N_STEPS = 1000    # enough to observe full escape dynamics
CURV_BETA = 0.1   # curvature correction strength

# Escape threshold: |y| starting from 0.5, reaching 2.0 means clear escape
ESCAPE_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Test functions (all 2D)
# ---------------------------------------------------------------------------

@dataclass
class TestFunction:
    name: str
    loss: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    description: str


def make_saddle_1():
    """L = x^2 - y^2, canonical saddle at origin."""
    def loss(p): return p[0]**2 - p[1]**2
    def grad(p): return np.array([2*p[0], -2*p[1]])
    def hess(p): return np.array([[2.0, 0.0], [0.0, -2.0]])
    return TestFunction("saddle_x2_y2", loss, grad, hess,
                        np.array([1.0, 0.5]), "L = x^2 - y^2")


def make_saddle_2():
    """L = x^2 - 3y^2, asymmetric saddle."""
    def loss(p): return p[0]**2 - 3*p[1]**2
    def grad(p): return np.array([2*p[0], -6*p[1]])
    def hess(p): return np.array([[2.0, 0.0], [0.0, -6.0]])
    return TestFunction("saddle_x2_3y2", loss, grad, hess,
                        np.array([1.0, 0.5]), "L = x^2 - 3y^2")


def make_nonconvex():
    """L = (x^2 - 1)^2 + x*y + y^2."""
    def loss(p):
        x, y = p[0], p[1]
        return (x**2 - 1)**2 + x*y + y**2
    def grad(p):
        x, y = p[0], p[1]
        return np.array([4*x*(x**2 - 1) + y, x + 2*y])
    def hess(p):
        x = p[0]
        return np.array([[12*x**2 - 4, 1.0], [1.0, 2.0]])
    return TestFunction("nonconvex_quartic", loss, grad, hess,
                        np.array([0.1, 0.5]),
                        "L = (x^2 - 1)^2 + x*y + y^2")


# ---------------------------------------------------------------------------
# Metric update mechanics
# ---------------------------------------------------------------------------

def sigma(s: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(s, -METRIC_CLIP - 1, METRIC_CLIP + 1))


def clip_s(s: np.ndarray) -> np.ndarray:
    s = s - np.mean(s)
    s = np.clip(s, -METRIC_CLIP, METRIC_CLIP)
    return s


def compute_denominator(s: np.ndarray, g: np.ndarray) -> float:
    return 1.0 / (1.0 + XI * np.sum(sigma(s) * g**2))


def compute_riemannian_hessian_eigs(H: np.ndarray, s: np.ndarray,
                                     g: np.ndarray) -> np.ndarray:
    """
    Eigenvalues of corrected Riemannian Hessian (H - C(s)) / denom.
    C_ii approximated as sign(H_ii) * sigma(s_i) * g_i^2.
    """
    denom = 1.0 + XI * np.sum(sigma(s) * g**2)
    H_diag = np.diag(H)
    correction = np.sign(H_diag) * sigma(s) * g**2
    H_corrected = H.copy()
    np.fill_diagonal(H_corrected, H_diag - correction)
    return np.linalg.eigvalsh(H_corrected / denom)


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRecord:
    theta: List[np.ndarray] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    s: List[np.ndarray] = field(default_factory=list)
    s_precond: List[np.ndarray] = field(default_factory=list)
    s_curv: List[np.ndarray] = field(default_factory=list)
    riem_eigs: List[np.ndarray] = field(default_factory=list)
    geodesic_convex: List[bool] = field(default_factory=list)
    y_speed: List[float] = field(default_factory=list)  # |dy/dt| proxy


# ---------------------------------------------------------------------------
# Optimizer variants
# ---------------------------------------------------------------------------

def _update_s_base(s, g):
    """Base metric update: s grows proportional to gradient magnitude."""
    s_grad = XI * sigma(s) * g**2
    return s + MU * (s_grad - LAMBDA_S * s)


def _update_s_escape(s, g, H_diag):
    """Escape-focused: s grows MORE where H < 0 (negative curvature)."""
    curv_sign = np.sign(H_diag)
    s_grad = XI * sigma(s) * g**2 - CURV_BETA * curv_sign * sigma(s) * g**2
    return s + MU * (s_grad - LAMBDA_S * s)


def _update_s_geodesic(s, g, H_diag):
    """Geodesic-focused: s grows MORE where H > 0 (positive curvature)."""
    curv_sign = np.sign(H_diag)
    s_grad = XI * sigma(s) * g**2 + CURV_BETA * curv_sign * sigma(s) * g**2
    return s + MU * (s_grad - LAMBDA_S * s)


def run_variant(fn: TestFunction, variant: str, n_steps: int = N_STEPS) -> TrajectoryRecord:
    """
    Run one of four variants:
      'baseline': single s, base rule
      'escape': single s, escape rule (-sign(H))
      'geodesic': single s, geodesic rule (+sign(H))
      'decoupled': two s vectors, escape for precond + geodesic for curv
    """
    rec = TrajectoryRecord()
    theta = fn.x0.copy()
    s = np.zeros(2)
    s_precond = np.zeros(2)
    s_curv = np.zeros(2)
    prev_theta = theta.copy()

    for t in range(n_steps):
        g = fn.grad(theta)
        H = fn.hessian(theta)
        L = fn.loss(theta)
        H_diag = np.diag(H)

        # Record
        rec.theta.append(theta.copy())
        rec.loss.append(L)

        if variant == "decoupled":
            rec.s.append(s_precond.copy())
            rec.s_precond.append(s_precond.copy())
            rec.s_curv.append(s_curv.copy())
            eigs = compute_riemannian_hessian_eigs(H, s_curv, g)
            active_s = s_precond
        else:
            rec.s.append(s.copy())
            eigs = compute_riemannian_hessian_eigs(H, s, g)
            active_s = s

        rec.riem_eigs.append(eigs)
        rec.geodesic_convex.append(bool(np.all(eigs >= 0)))

        # Speed: |delta_y| from last step
        if t > 0:
            rec.y_speed.append(abs(theta[1] - prev_theta[1]))
        else:
            rec.y_speed.append(0.0)
        prev_theta = theta.copy()

        # Parameter update
        if variant == "decoupled":
            r = compute_denominator(s_precond, g)
            theta = theta - LR * sigma(s_precond) * g * r
        else:
            r = compute_denominator(s, g)
            theta = theta - LR * sigma(s) * g * r

        # Metric update
        if variant == "baseline":
            s = clip_s(_update_s_base(s, g))
        elif variant == "escape":
            s = clip_s(_update_s_escape(s, g, H_diag))
        elif variant == "geodesic":
            s = clip_s(_update_s_geodesic(s, g, H_diag))
        elif variant == "decoupled":
            s_precond = clip_s(_update_s_escape(s_precond, g, H_diag))
            s_curv = clip_s(_update_s_geodesic(s_curv, g, H_diag))

    # Final record
    rec.theta.append(theta.copy())
    rec.loss.append(fn.loss(theta))
    if variant == "decoupled":
        rec.s.append(s_precond.copy())
        rec.s_precond.append(s_precond.copy())
        rec.s_curv.append(s_curv.copy())
    else:
        rec.s.append(s.copy())

    return rec


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def find_escape_step(rec: TrajectoryRecord, threshold: float = ESCAPE_THRESHOLD) -> Optional[int]:
    for i, th in enumerate(rec.theta):
        if abs(th[1]) > threshold:
            return i
    return None


def gc_fraction(rec: TrajectoryRecord) -> float:
    if not rec.geodesic_convex:
        return 0.0
    return sum(rec.geodesic_convex) / len(rec.geodesic_convex)


def max_y_speed_window(rec: TrajectoryRecord, window: int = 50) -> float:
    """Maximum average |dy/dt| over a sliding window."""
    if len(rec.y_speed) < window:
        return np.mean(rec.y_speed) if rec.y_speed else 0.0
    speeds = np.array(rec.y_speed)
    avgs = np.convolve(speeds, np.ones(window)/window, mode='valid')
    return float(np.max(avgs))


def s_divergence_time(rec: TrajectoryRecord, threshold: float = 1.0) -> Optional[int]:
    """Step at which |s_1 - s_2| first exceeds threshold."""
    for i, sv in enumerate(rec.s):
        if abs(sv[0] - sv[1]) > threshold:
            return i
    return None


def summarize_run(rec, variant, fn_name):
    esc = find_escape_step(rec)
    gc = gc_fraction(rec)
    ms = max_y_speed_window(rec)
    s_div = s_divergence_time(rec)
    final_y = abs(rec.theta[-1][1])

    result = {
        "variant": variant,
        "function": fn_name,
        "escape_step": esc,
        "gc_fraction": round(gc, 4),
        "final_loss": round(rec.loss[-1], 6),
        "final_theta": [round(x, 6) for x in rec.theta[-1]],
        "final_s": [round(x, 4) for x in rec.s[-1]],
        "final_abs_y": round(final_y, 4),
        "max_y_speed": round(ms, 6),
        "s_diverge_step": s_div,
    }
    if rec.s_precond:
        result["final_s_precond"] = [round(x, 4) for x in rec.s_precond[-1]]
        result["final_s_curv"] = [round(x, 4) for x in rec.s_curv[-1]]
    return result


def trajectory_to_dict(rec, step_interval=20):
    n = len(rec.theta)
    steps = list(range(0, n, step_interval))
    d = {
        "steps": steps,
        "theta": [[round(x, 6) for x in rec.theta[i]] for i in steps if i < n],
        "loss": [round(rec.loss[i], 6) for i in steps if i < len(rec.loss)],
        "s": [[round(x, 4) for x in rec.s[i]] for i in steps if i < len(rec.s)],
    }
    if rec.riem_eigs:
        d["riem_min_eig"] = [round(float(rec.riem_eigs[i].min()), 6)
                              for i in steps if i < len(rec.riem_eigs)]
    if rec.s_precond:
        d["s_precond"] = [[round(x, 4) for x in rec.s_precond[i]]
                           for i in steps if i < len(rec.s_precond)]
        d["s_curv"] = [[round(x, 4) for x in rec.s_curv[i]]
                        for i in steps if i < len(rec.s_curv)]
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    test_fns = [make_saddle_1(), make_saddle_2(), make_nonconvex()]
    variant_names = ["baseline", "escape", "geodesic", "decoupled"]

    all_results = []
    all_trajectories = {}

    print("=" * 80)
    print("DECOUPLED METRICS PROBE (C2-decoupled)")
    print("=" * 80)
    print(f"lr={LR}, mu={MU}, xi={XI}, lambda_s={LAMBDA_S}, "
          f"clip={METRIC_CLIP}, curv_beta={CURV_BETA}, steps={N_STEPS}")
    print(f"Escape threshold: |y| > {ESCAPE_THRESHOLD}")
    print()

    for fn in test_fns:
        H0 = fn.hessian(fn.x0)
        eigs0 = np.linalg.eigvalsh(H0)
        print("-" * 80)
        print(f"Function: {fn.description}")
        print(f"  x0={fn.x0}, H_diag={np.diag(H0)}, eigs={eigs0}")
        print()

        for vname in variant_names:
            t0 = time.time()
            rec = run_variant(fn, vname)
            dt = time.time() - t0

            summary = summarize_run(rec, vname, fn.name)
            all_results.append(summary)
            all_trajectories[f"{fn.name}__{vname}"] = trajectory_to_dict(rec)

            esc_str = f"step {summary['escape_step']}" if summary['escape_step'] is not None else "NEVER"
            gc_str = f"{summary['gc_fraction']*100:.1f}%"

            line = (f"  {vname:12s} | esc: {esc_str:10s} | gc: {gc_str:7s} "
                    f"| |y|={summary['final_abs_y']:8.3f} "
                    f"| max_dy={summary['max_y_speed']:.5f} "
                    f"| s=[{summary['final_s'][0]:+.2f},{summary['final_s'][1]:+.2f}]")

            if "final_s_precond" in summary:
                line += (f"  pre=[{summary['final_s_precond'][0]:+.2f},"
                         f"{summary['final_s_precond'][1]:+.2f}]"
                         f"  cur=[{summary['final_s_curv'][0]:+.2f},"
                         f"{summary['final_s_curv'][1]:+.2f}]")
            line += f"  ({dt:.2f}s)"
            print(line)

        print()

    # -----------------------------------------------------------------------
    # Deep dive: metric dynamics on saddle_x2_y2
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("DEEP DIVE: Metric dynamics on L = x^2 - y^2")
    print("=" * 80)
    print()
    print("The key observation: on a pure saddle, ALL variants develop s_1 >> s_2")
    print("initially because the gradient is larger in x (start at x=1, y=0.5).")
    print("The base drive xi * exp(s) * g^2 dominates early. The curvature")
    print("correction only matters once the x-gradient has decayed.")
    print()

    for vname in variant_names:
        key = f"saddle_x2_y2__{vname}"
        traj = all_trajectories[key]
        print(f"  {vname}:")
        for i, (step, th, sv) in enumerate(zip(traj["steps"], traj["theta"], traj["s"])):
            if step % 100 == 0:
                # Show effective step sizes
                s_arr = np.array(sv)
                sig = sigma(s_arr)
                print(f"    t={step:4d}: y={th[1]:+8.4f}, "
                      f"s=[{sv[0]:+.3f},{sv[1]:+.3f}], "
                      f"exp(s)=[{sig[0]:.4f},{sig[1]:.4f}]")
        print()

    # -----------------------------------------------------------------------
    # Decoupled deep dive: show s_precond vs s_curv divergence
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("DECOUPLED: s_precond vs s_curv trajectories")
    print("=" * 80)
    print()

    for fn in test_fns:
        key = f"{fn.name}__decoupled"
        traj = all_trajectories[key]
        if "s_precond" not in traj:
            continue
        print(f"  {fn.description}:")
        for i, step in enumerate(traj["steps"]):
            if i < len(traj.get("s_precond", [])) and step % 100 == 0:
                sp = traj["s_precond"][i]
                sc = traj["s_curv"][i]
                print(f"    t={step:4d}: s_precond=[{sp[0]:+.3f},{sp[1]:+.3f}]  "
                      f"s_curv=[{sc[0]:+.3f},{sc[1]:+.3f}]  "
                      f"delta=[{sp[0]-sc[0]:+.3f},{sp[1]-sc[1]:+.3f}]")
        print()

    # -----------------------------------------------------------------------
    # Cross-variant comparison table
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Function':<22s} {'Variant':<12s} {'Esc step':>10s} {'|y| final':>10s} "
          f"{'GC frac':>8s} {'max dy/dt':>10s}")
    print("-" * 80)

    for fn in test_fns:
        fn_results = [r for r in all_results if r["function"] == fn.name]
        for r in fn_results:
            esc_str = str(r["escape_step"]) if r["escape_step"] is not None else "never"
            print(f"{fn.name:<22s} {r['variant']:<12s} {esc_str:>10s} "
                  f"{r['final_abs_y']:>10.4f} "
                  f"{r['gc_fraction']*100:>7.1f}% "
                  f"{r['max_y_speed']:>10.6f}")
        print()

    # -----------------------------------------------------------------------
    # Key findings
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Per-function analysis
    for fn in test_fns:
        fn_res = {r["variant"]: r for r in all_results if r["function"] == fn.name}
        print(f"--- {fn.description} ---")
        print()

        # Escape comparison
        variants_that_escape = [v for v, r in fn_res.items()
                                if r["escape_step"] is not None]
        if variants_that_escape:
            print(f"  Variants that escape: {variants_that_escape}")
            fastest = min(variants_that_escape,
                          key=lambda v: fn_res[v]["escape_step"])
            print(f"  Fastest escape: {fastest} at step {fn_res[fastest]['escape_step']}")
        else:
            # Compare by final |y| as proxy for escape progress
            best_y = max(fn_res.items(), key=lambda kv: kv[1]["final_abs_y"])
            print(f"  No variant reaches escape threshold ({ESCAPE_THRESHOLD})")
            print(f"  Best escape progress (|y|): {best_y[0]} "
                  f"with |y|={best_y[1]['final_abs_y']:.4f}")

        # Geodesic convexity
        best_gc = max(fn_res.items(), key=lambda kv: kv[1]["gc_fraction"])
        print(f"  Best geodesic convexity: {best_gc[0]} "
              f"with {best_gc[1]['gc_fraction']*100:.1f}%")

        # Decoupled analysis
        if "decoupled" in fn_res:
            d = fn_res["decoupled"]
            e = fn_res.get("escape", {})
            g = fn_res.get("geodesic", {})

            # Does decoupled match escape variant in escape speed?
            if d.get("final_abs_y") and e.get("final_abs_y"):
                y_ratio = d["final_abs_y"] / max(e["final_abs_y"], 1e-10)
                print(f"  Decoupled |y| / Escape |y|: {y_ratio:.4f} "
                      f"(1.0 = identical escape progress)")

            # Does decoupled get better GC than escape?
            if e.get("gc_fraction") is not None:
                gc_diff = d["gc_fraction"] - e["gc_fraction"]
                print(f"  Decoupled GC - Escape GC: {gc_diff*100:+.1f}%")

            # Show s_precond vs s_curv divergence
            if "final_s_precond" in d and "final_s_curv" in d:
                sp = d["final_s_precond"]
                sc = d["final_s_curv"]
                print(f"  Metric divergence: s_precond={sp}, s_curv={sc}")
                print(f"    precond drives escape: s_precond_2 - s_precond_1 = "
                      f"{sp[1]-sp[0]:+.3f}")
                print(f"    curv drives geodesic: s_curv_1 - s_curv_2 = "
                      f"{sc[0]-sc[1]:+.3f}")

        print()

    # Final conclusions
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()

    # Gather saddle function results
    saddle_fns = [fn for fn in test_fns if "saddle" in fn.name]
    for fn in saddle_fns:
        fn_res = {r["variant"]: r for r in all_results if r["function"] == fn.name}

        dec = fn_res.get("decoupled", {})
        esc = fn_res.get("escape", {})
        geo = fn_res.get("geodesic", {})
        base = fn_res.get("baseline", {})

        # The fundamental question
        dec_escapes_faster = (dec.get("final_abs_y", 0) > esc.get("final_abs_y", 0))
        dec_more_gc = (dec.get("gc_fraction", 0) > esc.get("gc_fraction", 0))

    print("1. DOES DECOUPLING RESOLVE THE TENSION?")
    print()
    print("   The decoupled variant uses s_precond (escape-focused) for the")
    print("   parameter update and s_curv (geodesic-focused) for the curvature")
    print("   diagnostic. Since the parameter update ONLY uses s_precond, the")
    print("   escape dynamics are IDENTICAL to the escape-only variant.")
    print()
    print("   The benefit is that s_curv tracks geodesic convexity independently,")
    print("   providing a DIAGNOSTIC of whether the Riemannian Hessian is PSD")
    print("   even while the parameter update optimizes for escape.")
    print()

    print("2. ESCAPE SPEED COMPARISON")
    print()
    for fn in saddle_fns:
        fn_res = {r["variant"]: r for r in all_results if r["function"] == fn.name}
        speeds = [(v, r["final_abs_y"], r["max_y_speed"])
                  for v, r in fn_res.items()]
        speeds.sort(key=lambda x: -x[1])
        print(f"   {fn.description}:")
        for v, y, dy in speeds:
            print(f"     {v:12s}: |y|_final={y:.4f}, max_speed={dy:.6f}")
        print()

    print("3. GEODESIC CONVEXITY")
    print()
    for fn in saddle_fns:
        fn_res = {r["variant"]: r for r in all_results if r["function"] == fn.name}
        gcs = [(v, r["gc_fraction"]) for v, r in fn_res.items()]
        gcs.sort(key=lambda x: -x[1])
        print(f"   {fn.description}:")
        for v, gc in gcs:
            print(f"     {v:12s}: {gc*100:.1f}%")
        print()

    print("4. BENEFIT OVER ESCAPE-ONLY?")
    print()
    print("   For preconditioning: NONE. Decoupled s_precond = escape-only s.")
    print("   For diagnostics: YES. s_curv independently tracks geodesic convexity.")
    print("   For future use: s_curv could feed back into the learning rate")
    print("   schedule or step direction. When geodesic convexity is achieved,")
    print("   the optimizer knows it can trust the loss landscape geometry.")
    print()

    print("5. LIMITATIONS")
    print()
    print("   a) In this probe, s_curv is PURELY diagnostic — it does not affect")
    print("      the parameter trajectory. A full implementation would need to")
    print("      define HOW s_curv feeds back (e.g., modulate lr, trust region).")
    print("   b) The base gradient drive (xi * exp(s) * g^2) dominates the")
    print("      curvature correction early on. The correction only matters")
    print("      once gradients have similar magnitude in all directions.")
    print("   c) Mean-centering forces s_1 + s_2 = 0 (in 2D), so the metric")
    print("      can only differentiate RELATIVE scales, not absolute ones.")
    print("   d) The metric clip bounds |s_i| <= 4, which limits the condition")
    print("      number to exp(8) ~ 2981. This is a design choice.")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "probe": "C2-decoupled-metrics",
        "hyperparameters": {
            "lr": LR, "mu": MU, "xi": XI, "lambda_s": LAMBDA_S,
            "metric_clip": METRIC_CLIP, "n_steps": N_STEPS,
            "curv_beta": CURV_BETA, "escape_threshold": ESCAPE_THRESHOLD,
        },
        "results": all_results,
        "trajectories": all_trajectories,
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "decoupled_metrics_probe.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {os.path.abspath(out_path)}")

    return output


if __name__ == "__main__":
    run_all()
