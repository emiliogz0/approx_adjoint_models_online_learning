from pathlib import Path
import jax 
import jax.numpy as jnp

root_path = Path.cwd()

from configs import ModelState, Params

def rollout(
    step_fn,  
    state_init: ModelState,
    params: Params,
    nsteps: int,
    save_traj: bool = False,
):
    def scan_fn(state, _):
        next_state = step_fn(state, params) # Purely functional call
        return next_state, next_state.x

    final_state, xs = jax.lax.scan(scan_fn, state_init, None, length=nsteps)
    traj = jnp.concatenate([state_init.x[None], xs], axis=0) if save_traj else None
    return final_state, traj