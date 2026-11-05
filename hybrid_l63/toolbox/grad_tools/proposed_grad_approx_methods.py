import jax
import jax.numpy as jnp

from configs import ModelState
from toolbox.grad_tools.config_grad_approx import _anchor_global_indices
from toolbox.pytree_tools import params_partition, params_recombine, _add, _zeros
from toolbox.loss_tools import ensure_loss_cfg, build_obs_index

# ══════════════════════════════════════════════════════════════════════════════
#                  PULLBACK APPROXIMATION (PA) GRADIENT METHODS
#
# Factorized double-loop backward pass (Algorithm 1 of the paper). Each function
# is a complete, self-contained gradient computation with the same signature as
# the baselines in baseline_grad_methods.py -- no custom_vjp involved:
#
#   1. Forward rollout: N steps, keeping only the states (no tape).
#   2. Loss cotangents dL/du_n for every step, from one reverse pass through the
#      loss alone (the model is not differentiated here).
#   3. Outer loop over chunks (reverse): ONE pullback is constructed at the
#      chunk's anchor state, giving VJP^u_i and VJP^theta_i.
#      Inner loop over steps (reverse): inject the loss cotangent, accumulate the
#      adjoint into S_i, propagate the state adjoint (identity off-anchor for ZO,
#      frozen VJP^u_i for Reuse).
#      VJP^theta_i(S_i) is applied ONCE per chunk, after the inner loop.
#
# Chunk/anchor layout comes from GradApproxConfig (setup_grad_approx): n_chunks
# chunks, padded to max_k steps (padding index == nsteps), anchor at the middle
# of each chunk's real steps.
# ══════════════════════════════════════════════════════════════════════════════

def _pullback_approx_bptt(state_init, params, target, step_fn, loss_fn,
                          grad_approx_cfg, loss_eval, grad_mask, zero_order):
    if grad_mask is None:
        grad_mask = jax.tree_util.tree_map(lambda _: True, params)

    loss_cfg = ensure_loss_cfg(loss_eval)
    dynamic_p, static_p = params_partition(params, grad_mask)

    def step_fn_dynamic(s, d_p):
        return step_fn(s, params_recombine(d_p, static_p))

    nsteps       = grad_approx_cfg.nsteps
    step_indices = grad_approx_cfg.step_indices            # (n_chunks, max_k)
    anchor_idx   = _anchor_global_indices(grad_approx_cfg)  # (n_chunks,)
    is_obs, obs_idx, n_obs = build_obs_index(nsteps, loss_cfg.obs_stride)

    # ── 1. Forward rollout (states only) ──────────────────────────────────────
    def step_forward(state_curr, _):
        state_next = step_fn_dynamic(state_curr, dynamic_p)
        return state_next, state_next.x

    _, x_currs = jax.lax.scan(step_forward, state_init, None, length=nsteps)

    # ── 2. Loss and loss cotangents dL/du_n, n = 1..N ─────────────────────────
    def trajectory_loss(xs):
        if loss_cfg.mode == "terminal":
            return loss_fn(xs[-1], target[-1])
        per_step = jax.vmap(loss_fn)(xs, target[obs_idx])
        return jnp.sum(jnp.where(is_obs, per_step, 0.0)) / n_obs

    loss, dL_dx = jax.value_and_grad(trajectory_loss)(x_currs)

    # Anchor state = INPUT state of the anchor step (u_{n-1} for step n).
    anchor_states = jnp.where(
        (anchor_idx > 0)[:, None],
        x_currs[jnp.maximum(anchor_idx - 1, 0)],
        state_init.x[None],
    )

    # ── 3. Factorized double-loop backward pass ───────────────────────────────
    def outer_chunk_bwd(carry, xs):
        g, g_theta = carry
        chunk_steps, anc_step, anc_state = xs

        # Single pullback construction at the anchor -> VJP^u_i and VJP^theta_i
        _, pullback = jax.vjp(step_fn_dynamic, ModelState(x=anc_state, t=None), dynamic_p)
        vjp_state = lambda v: pullback(ModelState(x=v, t=None))[0].x
        vjp_param = lambda v: pullback(ModelState(x=v, t=None))[1]

        def inner_step_bwd(inner_carry, step_idx):
            g, S = inner_carry
            valid = step_idx < nsteps                     # False on padded slots
            safe_idx = jnp.minimum(step_idx, nsteps - 1)

            g_n = g + jnp.where(valid, dL_dx[safe_idx], 0.0)   # inject loss cotangent
            S = S + jnp.where(valid, g_n, 0.0)                 # accumulate for VJP^theta

            if zero_order:
                # identity off anchors (padded slots never match the anchor)
                g = jax.lax.cond(step_idx == anc_step, vjp_state, lambda v: v, g_n)
            else:
                # frozen anchor state VJP at every real step
                g = jnp.where(valid, vjp_state(g_n), g)
            return (g, S), None

        (g, S), _ = jax.lax.scan(
            inner_step_bwd, (g, jnp.zeros_like(g)), chunk_steps, reverse=True,
        )
        # Parameter VJP applied once per chunk
        return (g, _add(g_theta, vjp_param(S))), None

    (_, grads), _ = jax.lax.scan(
        outer_chunk_bwd,
        (jnp.zeros_like(state_init.x), _zeros(dynamic_p)),
        (step_indices, anchor_idx, anchor_states),
        reverse=True,
    )
    return loss, grads


# --------------- ZERO-ORDER ---------------------
def compute_zero_order(state_init, params, target, step_fn, loss_fn,
                       grad_approx_cfg, loss_eval="running", grad_mask=None):
    """
    Zero-order PA (Eq. 10): state pullback replaced by the identity between anchors,
    exact anchor pullback at the anchor step; parameter pullback linearized at the anchor.
    Forward steps : N + K      (K = n_chunks: one pullback construction per anchor)
    VJP evals     : K VJP^u + K VJP^theta
    Peak memory   : N states + 1 AD residual
    """
    return _pullback_approx_bptt(state_init, params, target, step_fn, loss_fn,
                                 grad_approx_cfg, loss_eval, grad_mask, zero_order=True)


# ------------------ REUSE ------------------------
def compute_reuse(state_init, params, target, step_fn, loss_fn,
                  grad_approx_cfg, loss_eval="running", grad_mask=None):
    """
    Pullback Reuse PA (Eq. 11): the anchor state and parameter pullbacks are frozen
    and reused at every step of the chunk.
    Forward steps : N + K      (K = n_chunks: one pullback construction per anchor)
    VJP evals     : N VJP^u + K VJP^theta
    Peak memory   : N states + 1 AD residual
    """
    return _pullback_approx_bptt(state_init, params, target, step_fn, loss_fn,
                                 grad_approx_cfg, loss_eval, grad_mask, zero_order=False)
