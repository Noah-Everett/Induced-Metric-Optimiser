"""
Comparison script to validate the mathematical equivalence between JAX and PyTorch
implementations of the off-diagonal induced-metric SGD optimizers.

This script tests that the JAX and PyTorch implementations produce identical results
when given the same inputs and initial conditions, for:

1. Mode-based a and b (using base_mode / a_mode / b_mode).
2. Static a_static / b_static (user-supplied fixed directions).
3. Dynamic a_fn / b_fn callbacks (user-defined functions of params/grads/momentum).
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any
import warnings

from jax_offdiag import custom_sgd_offdiag
from torch_offdiag import CustomSGDOffDiag


# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


def create_test_data(shape: Tuple[int, ...], key: jax.Array) -> Tuple[np.ndarray, torch.Tensor, jax.Array]:
    """Create test data in numpy, torch, and jax formats (same underlying values)."""
    np_data = np.random.randn(*shape).astype(np.float32)
    torch_data = torch.from_numpy(np_data)
    jax_data = jnp.array(np_data)
    return np_data, torch_data, jax_data


def torch_to_numpy(param_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert torch parameters dict to numpy dict."""
    return {k: v.detach().cpu().numpy() for k, v in param_dict.items()}


def jax_to_numpy(jax_dict: Dict[str, jax.Array]) -> Dict[str, np.ndarray]:
    """Convert jax parameters dict to numpy dict."""
    return {k: np.array(v) for k, v in jax_dict.items()}


def numpy_allclose(
    dict1: Dict[str, np.ndarray],
    dict2: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Check if all arrays in two dictionaries are close."""
    if set(dict1.keys()) != set(dict2.keys()):
        print(f"Keys don't match: {set(dict1.keys())} vs {set(dict2.keys())}")
        return False

    for key in dict1.keys():
        if not np.allclose(dict1[key], dict2[key], rtol=rtol, atol=atol):
            print(f"Arrays for key '{key}' don't match:")
            diff = dict1[key] - dict2[key]
            print(f"  Max absolute difference: {np.max(np.abs(diff))}")
            print(f"  Shape: {dict1[key].shape}")
            print(f"  JAX sample: {dict1[key].flat[:5]}")
            print(f"  PyTorch sample: {dict2[key].flat[:5]}")
            return False
    return True


# ------------------------------------------------------------------------------
# Test: off-diagonal modes (base_mode / a_mode / b_mode)
# ------------------------------------------------------------------------------

class TestOffDiagModes:
    """Test equivalence between JAX and PyTorch off-diag SGD using mode-based a, b."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        xi: float = 0.1,
        weight_decay: float = 0.0,
        gamma: float = 1.0,
        base_mode: str = "grad",
        a_mode: str = "momentum",
        b_mode: str = "same_as_a",
        use_momentum_for_update: bool = True,
    ):
        self.lr = lr
        self.momentum = momentum
        self.xi = xi
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.base_mode = base_mode
        self.a_mode = a_mode
        self.b_mode = b_mode
        self.use_momentum_for_update = use_momentum_for_update

    def test_single_run(self, param_shapes: List[Tuple[int, ...]], num_steps: int = 1) -> bool:
        """Run num_steps of optimization and compare final params."""
        key = set_seeds()

        # Create parameters and gradients
        params_jax: Dict[str, jax.Array] = {}
        params_torch: Dict[str, torch.Tensor] = {}
        grads_jax: Dict[str, jax.Array] = {}
        grads_torch: Dict[str, torch.Tensor] = {}

        for i, shape in enumerate(param_shapes):
            key, subkey = jax.random.split(key)
            np_param, torch_param, jax_param = create_test_data(shape, subkey)
            params_jax[f"param_{i}"] = jax_param
            params_torch[f"param_{i}"] = torch_param.clone().requires_grad_(True)

            key, subkey = jax.random.split(key)
            _, torch_grad, jax_grad = create_test_data(shape, subkey)
            grads_jax[f"param_{i}"] = jax_grad
            grads_torch[f"param_{i}"] = torch_grad

        # JAX optimizer
        jax_opt = custom_sgd_offdiag(
            learning_rate=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode=self.a_mode,
            b_mode=self.b_mode,
            use_momentum_for_update=self.use_momentum_for_update,
        )
        jax_state = jax_opt.init(params_jax)

        # PyTorch optimizer
        torch_opt = CustomSGDOffDiag(
            params_torch.values(),
            lr=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode=self.a_mode,
            b_mode=self.b_mode,
            use_momentum_for_update=self.use_momentum_for_update,
        )

        # Run optimization
        for _ in range(num_steps):
            # JAX
            updates_jax, jax_state = jax_opt.update(grads_jax, jax_state, params_jax)
            params_jax = jax.tree.map(lambda p, u: p + u, params_jax, updates_jax)

            # PyTorch
            torch_opt.zero_grad()
            # Assign fixed grads
            for i, p in enumerate(params_torch.values()):
                p.grad = grads_torch[f"param_{i}"]
            torch_opt.step()

        # Compare final params
        params_jax_np = jax_to_numpy(params_jax)
        params_torch_np = torch_to_numpy(params_torch)
        return numpy_allclose(params_jax_np, params_torch_np)


# ------------------------------------------------------------------------------
# Test: static a_static, b_static (user-specified fixed directions)
# ------------------------------------------------------------------------------

class TestOffDiagStaticAB:
    """Test equivalence when using static a_static / b_static."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        xi: float = 0.1,
        weight_decay: float = 0.0,
        gamma: float = 1.0,
        base_mode: str = "grad",
        use_momentum_for_update: bool = True,
    ):
        self.lr = lr
        self.momentum = momentum
        self.xi = xi
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.base_mode = base_mode
        self.use_momentum_for_update = use_momentum_for_update

    def test_single_run(self, param_shapes: List[Tuple[int, ...]], num_steps: int = 1) -> bool:
        key = set_seeds()

        params_jax: Dict[str, jax.Array] = {}
        params_torch: Dict[str, torch.Tensor] = {}
        grads_jax: Dict[str, jax.Array] = {}
        grads_torch: Dict[str, torch.Tensor] = {}

        # Build params and grads
        for i, shape in enumerate(param_shapes):
            key, subkey = jax.random.split(key)
            np_param, torch_param, jax_param = create_test_data(shape, subkey)
            params_jax[f"param_{i}"] = jax_param
            params_torch[f"param_{i}"] = torch_param.clone().requires_grad_(True)

            key, subkey = jax.random.split(key)
            _, torch_grad, jax_grad = create_test_data(shape, subkey)
            grads_jax[f"param_{i}"] = jax_grad
            grads_torch[f"param_{i}"] = torch_grad

        # Build static a and b (same shapes as params)
        a_static_jax: Dict[str, jax.Array] = {}
        b_static_jax: Dict[str, jax.Array] = {}
        a_static_torch_list: List[torch.Tensor] = []
        b_static_torch_list: List[torch.Tensor] = []

        for i, shape in enumerate(param_shapes):
            # a_static
            key, subkey = jax.random.split(key)
            _, torch_a, jax_a = create_test_data(shape, subkey)
            a_static_jax[f"param_{i}"] = jax_a
            a_static_torch_list.append(torch_a)

            # b_static
            key, subkey = jax.random.split(key)
            _, torch_b, jax_b = create_test_data(shape, subkey)
            b_static_jax[f"param_{i}"] = jax_b
            b_static_torch_list.append(torch_b)

        # JAX optimizer with static a / b
        jax_opt = custom_sgd_offdiag(
            learning_rate=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode="zero",      # ignored because a_static provided
            b_mode="same_as_a", # ignored because b_static provided
            use_momentum_for_update=self.use_momentum_for_update,
            a_static=a_static_jax,
            b_static=b_static_jax,
        )
        jax_state = jax_opt.init(params_jax)

        # PyTorch optimizer with static a / b
        torch_opt = CustomSGDOffDiag(
            params_torch.values(),
            lr=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode="zero",      # ignored
            b_mode="same_as_a", # ignored
            use_momentum_for_update=self.use_momentum_for_update,
            a_static=a_static_torch_list,
            b_static=b_static_torch_list,
        )

        # Run optimization
        for _ in range(num_steps):
            # JAX
            updates_jax, jax_state = jax_opt.update(grads_jax, jax_state, params_jax)
            params_jax = jax.tree.map(lambda p, u: p + u, params_jax, updates_jax)

            # PyTorch
            torch_opt.zero_grad()
            for i, p in enumerate(params_torch.values()):
                p.grad = grads_torch[f"param_{i}"]
            torch_opt.step()

        # Compare final params
        params_jax_np = jax_to_numpy(params_jax)
        params_torch_np = torch_to_numpy(params_torch)
        return numpy_allclose(params_jax_np, params_torch_np, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------------
# Test: dynamic a_fn, b_fn callbacks
# ------------------------------------------------------------------------------

class TestOffDiagFnAB:
    """Test equivalence when using dynamic a_fn / b_fn callbacks."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        xi: float = 0.1,
        weight_decay: float = 0.0,
        gamma: float = 1.0,
        base_mode: str = "grad",
        use_momentum_for_update: bool = True,
    ):
        self.lr = lr
        self.momentum = momentum
        self.xi = xi
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.base_mode = base_mode
        self.use_momentum_for_update = use_momentum_for_update

    def test_single_run(self, param_shapes: List[Tuple[int, ...]], num_steps: int = 1) -> bool:
        key = set_seeds()

        params_jax: Dict[str, jax.Array] = {}
        params_torch: Dict[str, torch.Tensor] = {}
        grads_jax: Dict[str, jax.Array] = {}
        grads_torch: Dict[str, torch.Tensor] = {}

        for i, shape in enumerate(param_shapes):
            key, subkey = jax.random.split(key)
            _, torch_param, jax_param = create_test_data(shape, subkey)
            params_jax[f"param_{i}"] = jax_param
            params_torch[f"param_{i}"] = torch_param.clone().requires_grad_(True)

            key, subkey = jax.random.split(key)
            _, torch_grad, jax_grad = create_test_data(shape, subkey)
            grads_jax[f"param_{i}"] = jax_grad
            grads_torch[f"param_{i}"] = torch_grad

        # Define JAX a_fn and b_fn (PyTree in -> PyTree out)
        def a_fn_jax(params, grads, m_hat, step):
            # Simple linear combination of p, g, m_hat
            return jax.tree.map(lambda p, g, m: 0.5 * p + 0.3 * g + 0.2 * m, params, grads, m_hat)

        def b_fn_jax(params, grads, m_hat, step):
            return jax.tree.map(lambda p, g, m: p - 0.25 * g + 0.1 * m, params, grads, m_hat)

        # JAX optimizer
        jax_opt = custom_sgd_offdiag(
            learning_rate=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode="zero",  # ignored when a_fn is set
            b_mode="same_as_a",  # ignored when b_fn is set
            use_momentum_for_update=self.use_momentum_for_update,
            a_fn=a_fn_jax,
            b_fn=b_fn_jax,
        )
        jax_state = jax_opt.init(params_jax)

        # Define PyTorch a_fn and b_fn with matching semantics
        def a_fn_torch(params_with_grad, grads, m_hat_list, step):
            return [
                0.5 * p.data + 0.3 * g + 0.2 * m
                for p, g, m in zip(params_with_grad, grads, m_hat_list)
            ]

        def b_fn_torch(params_with_grad, grads, m_hat_list, step):
            return [
                p.data - 0.25 * g + 0.1 * m
                for p, g, m in zip(params_with_grad, grads, m_hat_list)
            ]

        # PyTorch optimizer
        torch_opt = CustomSGDOffDiag(
            params_torch.values(),
            lr=self.lr,
            momentum=self.momentum,
            xi=self.xi,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            base_mode=self.base_mode,
            a_mode="zero",  # ignored
            b_mode="same_as_a",  # ignored
            use_momentum_for_update=self.use_momentum_for_update,
            a_fn=a_fn_torch,
            b_fn=b_fn_torch,
        )

        # Run optimization
        for _ in range(num_steps):
            # JAX
            updates_jax, jax_state = jax_opt.update(grads_jax, jax_state, params_jax)
            params_jax = jax.tree.map(lambda p, u: p + u, params_jax, updates_jax)

            # PyTorch
            torch_opt.zero_grad()
            for i, p in enumerate(params_torch.values()):
                p.grad = grads_torch[f"param_{i}"]
            torch_opt.step()

        # Compare final params
        params_jax_np = jax_to_numpy(params_jax)
        params_torch_np = torch_to_numpy(params_torch)
        return numpy_allclose(params_jax_np, params_torch_np, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------------
# Top-level runners
# ------------------------------------------------------------------------------

def run_mode_tests() -> bool:
    print("\n" + "=" * 80)
    print("OFF-DIAG MODES TESTS (base_mode / a_mode / b_mode)")
    print("=" * 80)

    test_shapes = [
        [(10,)],
        [(5, 5)],
        [(10,), (5, 5)],
        [(3, 4, 5)],
    ]

    configs = [
        dict(lr=0.01, momentum=0.9, xi=0.1, weight_decay=0.0, gamma=1.0,
             base_mode="grad", a_mode="momentum", b_mode="same_as_a", use_momentum_for_update=True),
        dict(lr=0.01, momentum=0.9, xi=0.2, weight_decay=0.01, gamma=0.5,
             base_mode="momentum", a_mode="grad", b_mode="grad", use_momentum_for_update=True),
        dict(lr=0.001, momentum=0.0, xi=0.1, weight_decay=0.0, gamma=1.0,
             base_mode="grad", a_mode="zero", b_mode="zero", use_momentum_for_update=False),
    ]

    all_passed = True

    for ci, cfg in enumerate(configs):
        for si, shapes in enumerate(test_shapes):
            tester = TestOffDiagModes(**cfg)
            print(f"\nConfig {ci+1}, Shapes {si+1}: ", end="")
            passed_1 = tester.test_single_run(shapes, num_steps=1)
            passed_3 = tester.test_single_run(shapes, num_steps=3)
            if passed_1 and passed_3:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                all_passed = False

    return all_passed


def run_static_tests() -> bool:
    print("\n" + "=" * 80)
    print("OFF-DIAG STATIC a_static / b_static TESTS")
    print("=" * 80)

    test_shapes = [
        [(10,)],
        [(5, 5)],
        [(10,), (5, 5)],
    ]

    configs = [
        dict(lr=0.01, momentum=0.9, xi=0.05, weight_decay=0.01, gamma=1.0,
             base_mode="grad", use_momentum_for_update=True),
        dict(lr=0.001, momentum=0.95, xi=0.1, weight_decay=0.0, gamma=0.75,
             base_mode="momentum", use_momentum_for_update=True),
    ]

    all_passed = True

    for ci, cfg in enumerate(configs):
        for si, shapes in enumerate(test_shapes):
            tester = TestOffDiagStaticAB(**cfg)
            print(f"\nConfig {ci+1}, Shapes {si+1}: ", end="")
            passed_1 = tester.test_single_run(shapes, num_steps=1)
            passed_3 = tester.test_single_run(shapes, num_steps=3)
            if passed_1 and passed_3:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                all_passed = False

    return all_passed


def run_function_tests() -> bool:
    print("\n" + "=" * 80)
    print("OFF-DIAG FUNCTIONAL a_fn / b_fn TESTS")
    print("=" * 80)

    test_shapes = [
        [(10,)],
        [(5, 5)],
        [(10,), (5, 5)],
    ]

    configs = [
        dict(lr=0.01, momentum=0.9, xi=0.1, weight_decay=0.0, gamma=1.0,
             base_mode="grad", use_momentum_for_update=True),
        dict(lr=0.01, momentum=0.9, xi=0.2, weight_decay=0.01, gamma=0.5,
             base_mode="momentum", use_momentum_for_update=True),
    ]

    all_passed = True

    for ci, cfg in enumerate(configs):
        for si, shapes in enumerate(test_shapes):
            tester = TestOffDiagFnAB(**cfg)
            print(f"\nConfig {ci+1}, Shapes {si+1}: ", end="")
            passed_1 = tester.test_single_run(shapes, num_steps=1)
            passed_3 = tester.test_single_run(shapes, num_steps=3)
            if passed_1 and passed_3:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                all_passed = False

    return all_passed


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore", category=UserWarning)

        print("Starting off-diagonal optimizer equivalence tests...")
        print("This may take a few moments...\n")

        modes_ok = run_mode_tests()
        static_ok = run_static_tests()
        fn_ok = run_function_tests()

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        if modes_ok and static_ok and fn_ok:
            print("🎉 ALL OFF-DIAG VERIFICATION TESTS PASSED! 🎉")
            print("\n✅ JAX and PyTorch off-diagonal optimizers are numerically consistent")
            print("✅ Modes, static a/b, and functional a_fn/b_fn all agree")
        else:
            print("❌ SOME OFF-DIAG TESTS FAILED")
            print("\n⚠️  There are discrepancies between implementations")
            print("⚠️  Review the failed tests above for details")

        print("=" * 80)

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure jax_offdiag.py and torch_offdiag.py are in the same directory")
        print("and that JAX and PyTorch are installed.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
