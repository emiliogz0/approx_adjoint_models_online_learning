import numpy as np
import jax
import jax.numpy as jnp

from configs import ModelState
from toolbox.grad_tools.config_grad_approx import GradApproxConfig, setup_grad_approx
from toolbox.pytree_tools import params_partition, params_recombine, _add, _zeros
from toolbox.loss_tools import ensure_loss_cfg, build_obs_index

# ══════════════════════════════════════════════════════════════════════════════
#                        BASELINE GRADIENT METHODS
#
# Each function below is a complete, self-contained gradient computation --
# forward and backward are both wired up right here via jax.value_and_grad.
# The approximate (Zero-order / Reuse) methods, with the same signature, live in
# toolbox/grad_tools/proposed_grad_approx_methods.py.
# ══════════════════════════════════════════════════════════════════════════════

def compute_exact_bptt(state_init, params, target, step_fn, loss_fn,
                       grad_approx_cfg, loss_eval="running", grad_mask=None):
    """
    Standard BPTT — JAX autodiff handles everything.
    Forward: N steps, saving all VJP closures.
    Backward: N VJP applications using pre-saved closures.
    Memory: O(N) VJP closures held simultaneously.
    """

    if grad_mask is None:
        grad_mask = jax.tree_util.tree_map(lambda _: True, params)

    loss_cfg = ensure_loss_cfg(loss_eval)
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    def objective_functional(d_p):

        def step_forward(state_curr, _):
            state_next = step_fn_dynamic(state_curr, d_p)
            return state_next, state_next.x

        state_final, x_currs = jax.lax.scan(
            step_forward, state_init, None, length=grad_approx_cfg.nsteps
        )

        if loss_cfg.mode == "terminal":
            return loss_fn(state_final.x, target[-1])
        else:
            is_obs, obs_idx, n_obs = build_obs_index(grad_approx_cfg.nsteps, loss_cfg.obs_stride)
            per_step = jax.vmap(loss_fn)(x_currs, target[obs_idx])
            return jnp.sum(jnp.where(is_obs, per_step, 0.0)) / n_obs

    loss, grads = jax.value_and_grad(objective_functional)(dynamic_p)
    return loss, grads

# ------------------     TRUNCATED BACKPROPAGATION   ------------------------ #
def compute_tbptt(state_init, params, target, step_fn, loss_fn,
                  grad_approx_cfg, loss_eval="running", grad_mask=None):
    """
    Standard TBPTT 
    Each chunk is treated as a local BPTT. Forward passes and with-in chunk pullbacks are interleaved.
    Memory: O(K) VJP closures per chunk, discarded after each chunk pullback.
    Terminal loss: only last chunk contributes — effective horizon K = N/A.
    Running loss:  all chunks contribute local loss gradients.
    """
    if grad_mask is None:
        grad_mask = jax.tree_util.tree_map(lambda _: True, params)

    loss_cfg = ensure_loss_cfg(loss_eval)
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    nsteps       = grad_approx_cfg.nsteps
    n_chunks     = grad_approx_cfg.n_chunks
    max_k        = grad_approx_cfg.max_k
    step_indices = grad_approx_cfg.step_indices  # (n_chunks, max_k)
    is_obs, obs_idx, n_obs = build_obs_index(nsteps, loss_cfg.obs_stride)


    if loss_cfg.mode == "running":
        step_idx_np = np.arange(nsteps)
        is_obs_np = (step_idx_np + 1) % loss_cfg.obs_stride == 0
        step_indices_np = np.asarray(step_indices)
        empty_chunks = [
            i for i, row in enumerate(step_indices_np)
            if not is_obs_np[row[row < nsteps]].any()
        ]
        if empty_chunks:
            print(
                f"TBPTT: chunk(s) {empty_chunks} of {n_chunks} contain no observation "
                f"(nsteps={nsteps}, obs_stride={loss_cfg.obs_stride}) -- TBPTT truncates "
                f"gradient flow at chunk boundaries, so each of those chunks contributes "
                f"exactly zero gradient of its own. Still running with this chunk layout."
            )

    def process_chunk(carry, xs):
        boundary_x, dL_dp_accum, loss_accum, chunk_idx = carry
        chunk_step_indices = xs

        boundary_state = ModelState(x=boundary_x, t=None)
        is_last_chunk  = (chunk_idx == n_chunks - 1)

        def chunk_loss_and_state(d_p):
            def inner_fwd(s, step_idx):
                valid  = step_idx < nsteps
                s_next = step_fn_dynamic(s, d_p)
                return ModelState(x=jax.lax.select(valid, s_next.x, s.x), t=None), s_next.x

            final_state, x_currs = jax.lax.scan(
                inner_fwd, boundary_state, chunk_step_indices, length=max_k,
            )

            if loss_cfg.mode == "terminal":
                loss = jax.lax.cond(
                    is_last_chunk,
                    lambda _: loss_fn(final_state.x, target[-1]),
                    lambda _: jnp.zeros(()),
                    operand=None,
                )
            else:
                def step_loss(xc, step_idx):
                    valid       = step_idx < nsteps
                    safe_idx    = jnp.minimum(step_idx, nsteps - 1)
                    contributes = valid & is_obs[safe_idx]
                    return jax.lax.select(
                        contributes,
                        loss_fn(xc, target[obs_idx[safe_idx]]) / n_obs,
                        jnp.zeros(()),
                    )
                loss = jnp.sum(jax.vmap(step_loss)(x_currs, chunk_step_indices))

            return loss, final_state.x

        (chunk_loss, final_x), chunk_grad = jax.value_and_grad(
            chunk_loss_and_state, has_aux=True
        )(dynamic_p)

        return (
            final_x,                              # boundary for next chunk
            _add(dL_dp_accum, chunk_grad),        # accumulated parameter grads
            loss_accum + chunk_loss,              # accumulated loss
            chunk_idx + 1,
        ), None

    (_, grads, loss, _), _ = jax.lax.scan(
        process_chunk,
        (state_init.x, _zeros(dynamic_p), jnp.zeros(()), jnp.int32(0)),
        step_indices,
    )

    return loss, grads

# ---------------------- CHECKPOINTING ----------------------------------------------------
def _setup_uniform_sqrt_checkpointing(nsteps: int, bias_dir: str = "upstream") -> GradApproxConfig:
    """Convenience wrapper: sets n_anchors = ceil(sqrt(nsteps)) for Griewank checkpointing."""
    n_chunks = int(jnp.ceil(jnp.sqrt(nsteps)))
    return setup_grad_approx(nsteps=nsteps, n_anchors=n_chunks, bias_dir=bias_dir)

def compute_uniform_sqrt_checkpointed_bptt(state_init, params, target, step_fn, loss_fn,
                                    grad_approx_cfg, loss_eval="running", grad_mask=None):
    """

    Exact gradient with O(sqrt(N)) peak memory — Chen 2016. checkpointing.
    Total step_fn forward calls : 2N  (N phase-1 + N phase-2 replays)
    VJP applications            : N   (exact — one per step, inside segment backwards)
    Peak memory                 : A checkpoint states  +  K live states  +  K in-step residuals
                                  ≈  2*sqrt(N) states  +  sqrt(N) traces
    Gradient                    : exact  (identical to compute_exact_bptt)

    Usage:
        cfg = setup_sqrt_checkpointing(nsteps=N)
        loss, grads = compute_checkpointed_bptt(..., grad_approx_cfg=cfg)
    """
    if grad_mask is None:
        grad_mask = jax.tree_util.tree_map(lambda _: True, params)

    loss_cfg = ensure_loss_cfg(loss_eval)
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    nsteps       = grad_approx_cfg.nsteps
    n_chunks   = grad_approx_cfg.n_chunks
    max_k        = grad_approx_cfg.max_k
    step_indices = grad_approx_cfg.step_indices   # (n_chunks, max_k)
    is_obs, obs_idx, n_obs = build_obs_index(nsteps, loss_cfg.obs_stride)

    def fwd_chunk(state_carry, seg_indices):  # Forward sweep — store only A checkpoint states
        def inner_step(s, step_idx):
            valid  = step_idx < nsteps
            s_next = step_fn_dynamic(s, dynamic_p)
            return ModelState(x=jax.lax.select(valid, s_next.x, s.x), t=None), None

        final_state, _ = jax.lax.scan(inner_step, state_carry, seg_indices, length=max_k)
        return final_state, state_carry.x   # checkpoint = START state of this segment

    x_N_state, checkpoint_xs = jax.lax.scan(fwd_chunk, state_init, step_indices)     # checkpoint_xs : (n_chunk, state_dim)

    # ── Initial boundary cotangent ────────────────────────────────────────
    if loss_cfg.mode == "terminal":
        terminal_loss, g_init = jax.value_and_grad(lambda x: loss_fn(x, target[-1]))(x_N_state.x)
    else:
        terminal_loss = jnp.zeros(())
        g_init = jnp.zeros_like(state_init.x)   

    def bwd_segment(carry, xs):   # Backward sweep — exact BPTT per chunk
        g, dL_dp_accum, loss_accum = carry
        chunk_indx, chk_x = xs

        def chunk_value(start_x, d_p):
            """
            grad w.r.t. start_x  →  outgoing cotangent for the previous segment.
            grad w.r.t. d_p      →  parameter gradient from this segment's contribution.
            The dot product injects the incoming cotangent g across the segment boundary,
            making the overall gradient exact despite processing one segment at a time.
            """
            start_state = ModelState(x=start_x, t=None)

            def inner_step(s, step_idx):
                valid  = step_idx < nsteps
                s_next = step_fn_dynamic(s, d_p)
                s_frozen = ModelState(x=jax.lax.select(valid, s_next.x, s.x), t=None)
                return s_frozen, (s_next.x, step_idx)

            final_s, (x_currs, step_idxs) = jax.lax.scan(
                inner_step, start_state, chunk_indx, length=max_k
            )

            if loss_cfg.mode == "terminal":
                seg_loss = jnp.zeros(())
            else:
                def step_l(xc, idx):
                    valid       = idx < nsteps
                    safe_idx    = jnp.minimum(idx, nsteps - 1)
                    contributes = valid & is_obs[safe_idx]
                    return jax.lax.select(contributes,
                        loss_fn(xc, target[obs_idx[safe_idx]]) / n_obs,
                        jnp.zeros(()))
                seg_loss = jnp.sum(jax.vmap(step_l)(x_currs, step_idxs))

            return seg_loss + jnp.dot(g, final_s.x), seg_loss

        (_, seg_loss), (g_new, d_p_seg) = jax.value_and_grad(
            chunk_value, argnums=(0, 1), has_aux=True
        )(chk_x, dynamic_p)

        return (
            g_new,
            _add(dL_dp_accum, d_p_seg),
            loss_accum + seg_loss,
        ), None

    (_, grads, loss), _ = jax.lax.scan(
        bwd_segment,
        (g_init, _zeros(dynamic_p), jnp.zeros(())),
        (step_indices, checkpoint_xs),
        reverse=True,
    )

    return loss + terminal_loss, grads

# ------------------------------------------------------------------------------------ #
def compute_recursive_binomal_checkpointed_bptt(state_init, params, target, step_fn, loss_fn,
                                    grad_approx_cfg, loss_eval="running", grad_mask=None):
    """
    Treeverse (Griewank. 19992)
    Very close to the revolve optimal checkpoint schedule but chunks always disected in the middle.
    In contrary, Revolve doesnt always split in half
    ("Achieving logarithmic growth of temporal and spatial complexity in reverse AD").
    Recursively bisects the timeline: to backprop through [i, j), replay forward from the
    checkpoint x_i to the midpoint, recurse on the right half (deeper) first, then recurse
    on the left half re-using x_i (already held on the call stack — no extra storage).

    Total step_fn forward calls : O(N log N)  (every recursion level replays ~N/2 steps total)
    VJP applications            : N           (one per leaf, exact — same as compute_exact_bptt)
    Peak memory                 : O(log N) checkpoint states held on the recursion stack
                                  +  1 live VJP closure (leaves are processed one at a time)
    Gradient                    : exact  (identical to compute_exact_bptt)
    """
    if grad_mask is None:
        grad_mask = jax.tree_util.tree_map(lambda _: True, params)

    loss_cfg = ensure_loss_cfg(loss_eval)
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    nsteps = grad_approx_cfg.nsteps

    _, _, n_obs = build_obs_index(nsteps, loss_cfg.obs_stride)

    def _replay_forward(x0, n_steps):
        """Recompute the state n_steps ahead of x0 — forward-only, no tape."""
        def step(s, _):
            return step_fn_dynamic(s, dynamic_p), None
        final_state, _ = jax.lax.scan(step, ModelState(x=x0, t=None), None, length=n_steps)
        return final_state.x

    def _leaf(x_i, i, g_next):
        """Exact local VJP for the single step i -> i+1, with injected future cotangent g_next."""
        def value_fn(start_x, d_p):
            s_next = step_fn_dynamic(ModelState(x=start_x, t=None), d_p)
            if loss_cfg.mode == "running" and (i + 1) % loss_cfg.obs_stride == 0:
                obs_i = (i + 1) // loss_cfg.obs_stride
                local_loss = loss_fn(s_next.x, target[obs_i]) / n_obs
            else:
                local_loss = jnp.zeros(())
            return local_loss + jnp.dot(g_next, s_next.x), local_loss

        (_, local_loss), (g_i, dpar) = jax.value_and_grad(
            value_fn, argnums=(0, 1), has_aux=True
        )(x_i, dynamic_p)
        return g_i, dpar, local_loss

    def _treeverse(x_i, i, j, g_j):
        """Exact adjoint over [i, j), given true state x_i and incoming cotangent g_j = dL/dx_j."""
        if j - i == 1:
            return _leaf(x_i, i, g_j)

        mid   = i + (j - i) // 2
        x_mid = _replay_forward(x_i, mid - i)

        g_mid, dpar_right, loss_right = _treeverse(x_mid, mid, j, g_j)
        g_i,   dpar_left,  loss_left  = _treeverse(x_i,   i,   mid, g_mid)

        return g_i, _add(dpar_left, dpar_right), loss_left + loss_right

    if loss_cfg.mode == "terminal":
        x_N = _replay_forward(state_init.x, nsteps)
        loss, g_N = jax.value_and_grad(lambda x: loss_fn(x, target[-1]))(x_N)
    else:
        g_N = jnp.zeros_like(state_init.x)

    _, grads, running_loss = _treeverse(state_init.x, 0, nsteps, g_N)

    if loss_cfg.mode == "running":
        loss = running_loss

    return loss, grads
