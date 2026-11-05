import jax
import jax.numpy  as jnp
import numpy as np 
from typing import Any 
import time 
from toolbox.pytree_tools import params_recombine, params_partition


# ── Metric functions ──────────────────────────────────────────────────────────

# --------- Quality of the gradients ---------------------
def _flat(g: Any) -> jnp.ndarray:
    flat, _ = jax.flatten_util.ravel_pytree(g)
    return flat

def grad_norm(g: Any) -> float:
    return float(jnp.linalg.norm(_flat(g)))

def relative_l2_error(g_ref: Any, g_test: Any) -> float:
    f_ref, f_test = _flat(g_ref), _flat(g_test)
    return float(jnp.linalg.norm(f_ref - f_test) / (jnp.linalg.norm(f_ref) + 1e-12))

def cosine_similarity(g_ref: Any, g_test: Any) -> float:
    f_ref, f_test = _flat(g_ref), _flat(g_test)
    return float(jnp.dot(f_ref, f_test) /
                 (jnp.linalg.norm(f_ref) * jnp.linalg.norm(f_test) + 1e-12))

def xla_flops(compiled_obj) -> float:
    """Extract XLA's FLOP estimate from an already compiled executable."""
    try:
        c = compiled_obj.cost_analysis()
        if isinstance(c, list):
            return float(sum(x.get('flops', 0.0) for x in c))
        return float(c.get('flops', float('nan')))
    except Exception:
        return float('nan')


def xla_estimated_flops(compiled_obj) -> float:
    """XLA cost-model estimate of FLOPs for the complete compiled computation.

    It is derived after XLA optimization, including fusion and dead-code
    elimination.  It is not a hardware-counter measurement of executed
    floating-point instructions.
    """
    return xla_flops(compiled_obj)


def xla_estimated_buffer_mb(compiled_obj) -> float:
    """XLA-estimated compiled-program buffer requirement, in MB.

    XLA computes it after compilation and buffer assignment.  It includes
    argument, output, and temporary buffers, subtracting aliases to avoid
    double-counting.  It excludes Python/JAX runtime overhead, compilation
    memory, and allocator-reserved capacity, so it is neither process RSS nor
    the GPU allocator's reserved capacity.
    """
    def _peak_bytes(stats, prefix=""):
        return (
            getattr(stats, f"{prefix}argument_size_in_bytes", 0)
            + getattr(stats, f"{prefix}output_size_in_bytes", 0)
            + getattr(stats, f"{prefix}temp_size_in_bytes", 0)
            - getattr(stats, f"{prefix}alias_size_in_bytes", 0)
        )

    try:
        stats = compiled_obj.memory_analysis()
        device_peak_bytes = _peak_bytes(stats)
        # Some CPU backends report their buffers through the host fields rather
        # than the default device fields.  Use those only as a fallback: on an
        # accelerator they describe a separate host-memory space.
        peak_bytes = device_peak_bytes or _peak_bytes(stats, "host_")
        return float(peak_bytes) / 1e6 if peak_bytes > 0 else float('nan')
    except Exception:
        return float('nan')

def wall_clock_ms(compiled_fn, *args) -> float:
    """Single timed call. Flush async dispatch first, then time one execution."""
    jax.block_until_ready(compiled_fn(*args))
    t0 = time.perf_counter()
    jax.block_until_ready(compiled_fn(*args))
    return (time.perf_counter() - t0) * 1e3


def xla_temp_bytes(compiled_obj) -> float:
    """XLA temporary-buffer bytes of a compiled executable (peak working memory,
    excluding arguments/outputs). Used for the Peak Memory Ratio of the fidelity benchmark."""
    try:
        return float(compiled_obj.memory_analysis().temp_size_in_bytes)
    except Exception:
        return float('nan')


def time_call_min(compiled_fn, *args, repeats=15, warmup=3):
    """Returns (output, best wall time in s) over `repeats` synchronized calls,
    after `warmup` untimed calls."""
    for _ in range(warmup):
        jax.block_until_ready(compiled_fn(*args))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = compiled_fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, float(np.min(times))


def estimate_step_flops(step_fn, state_example, params, grad_mask):
    """FLOPs of one forward model step."""
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    compiled = (jax.jit(step_fn_dynamic).lower(state_example, dynamic_p).compile())
    flops = xla_flops(compiled)
    print("── Atomic FLOPs ─────────────────────────────────────────────")
    print(f"  model step : {flops:.3e} FLOPs")
    print("─────────────────────────────────────────────────────────────")
    return flops

def estimate_atomic_bytes(step_fn, state_example, params, grad_mask):
    """
    Measures the two atomic memory units used in the theoretical models:
      - one model state
      - one pullback (VJP closure)
    """
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    def _count(obj):
        return sum(l.nbytes for l in jax.tree_util.tree_leaves(obj) if hasattr(l, "nbytes"))

    bytes_state = _count(state_example)
    _, pullback = jax.vjp(step_fn_dynamic, state_example, dynamic_p)
    bytes_pullback = _count(pullback)

    print("── Atomic Memory ────────────────────────────────────────────")
    print(f"  model state     : {bytes_state/1e3:.3f} KB")
    print(f"  pullback        : {bytes_pullback/1e3:.3f} KB")
    print("─────────────────────────────────────────────────────────────")

    return bytes_state, bytes_pullback

def theoretical_resource_counts(method, N, A):
    """Hand-counted resources; excludes pullback *application* cost.

    `step_evals` counts step_fn primals, including the primal evaluation used
    to construct a VJP. `states` counts explicitly retained ModelState-sized
    arrays in these implementations; it is intentionally separate from XLA's
    physical buffer allocation and may be reduced by compiler reuse.
    """
    K = int(np.ceil(N / A))
    specs = {
        'exact_bptt':                    (N,     N,     N),
        'tbptt':                         (N,     K,     K),
        # Pullback Approximation (double loop): N forward steps + A anchor pullback
        # constructions; x_currs plus A gathered anchor states; one live pullback.
        'zero_order':                    (N + A, N + A, 1),
        'reuse':                         (N + A, N + A, 1),
        # Named to match compute_uniform_sqrt_checkpointed_bptt /
        # compute_recursive_binomal_checkpointed_bptt in baseline_grad_methods.py --
        # 'checkpointed_bptt'/'log_checkpointed_bptt' were stale aliases that
        # never matched what METHOD_ROUTING_MAP/_build_grad_cfg actually pass,
        # so theoretical_flops/theoretical_peak_memory_mb raised ValueError
        # for both real method names.
        'uniform_sqrt_chkpt_bptt':       (2 * N, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))),
        'treeverse_chkpt_bptt':          (int(N * (1 + np.ceil(np.log2(max(N, 2))) / 2)), int(np.ceil(np.log2(max(N, 2)))) + 1, 1),
    }
    if method not in specs:
        raise ValueError(f'Unknown method {method!r}')
    step_evals, n_states, n_pullbacks = specs[method]
    return {'step_evals': step_evals, 'states': n_states, 'pullbacks': n_pullbacks}

def theoretical_flops(method, N, A, flops_step):
    """Proxy FLOPs: step_fn primal work only; excludes pullback applications."""

    return theoretical_resource_counts(method, N, A)['step_evals'] * flops_step

def theoretical_peak_memory_mb(method,N,A,bytes_state,bytes_pullback):
    """ Peak memory from the algorithmic complexity table."""

    counts = theoretical_resource_counts(method, N, A)
    n_states, n_pullbacks = counts['states'], counts['pullbacks']

    return (n_states * bytes_state + n_pullbacks * bytes_pullback) / 1e6
