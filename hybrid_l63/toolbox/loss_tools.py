import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from typing import Union


LOSS_MODES = ("running", "terminal")

@dataclass(frozen=True)
class LossConfig:
    """
    mode:
        "running"  -- every predicted step is compared against the matching
                      target step (target[1:]); loss is the mean over the
                      full horizon.
        "terminal" -- only the final state is compared against target[-1];
                      intermediate steps contribute nothing.
    obs_stride:
        In "running" mode, only steps that fall on an observation are scored --
        model step_idx (0-based; step_idx+1 steps completed) counts iff
        (step_idx+1) % obs_stride == 0, matched against
        target[(step_idx+1) // obs_stride]. obs_stride=1 (default) scores every
        step, recovering the original dense-loss behavior. Ignored in "terminal"
        mode. 
    """
    mode: str = "running"
    obs_stride: int = 1

    def __post_init__(self):
        if self.mode not in LOSS_MODES:
            raise ValueError(f"Unknown LossConfig.mode {self.mode!r}; expected one of {LOSS_MODES}")
        if self.obs_stride < 1:
            raise ValueError(f"LossConfig.obs_stride must be >= 1, got {self.obs_stride}")


def ensure_loss_cfg(loss_eval: Union[str, LossConfig]) -> LossConfig:
    """Accepts a bare 'running'/'terminal' string (legacy call sites) or an
    explicit LossConfig, and always returns a LossConfig."""
    if isinstance(loss_eval, LossConfig):
        return loss_eval
    return LossConfig(mode=loss_eval)


def loss_fn(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predicted - target) ** 2)


def build_obs_index(nsteps: int, obs_stride: int):
    """Maps each model step to whether it lands on an observation and, if so,
    which entry of the (sparse) target array it corresponds to.
    """
   
    step_idx = np.arange(nsteps)
    is_obs = (step_idx + 1) % obs_stride == 0
    obs_idx = (step_idx + 1) // obs_stride
    n_obs = nsteps // obs_stride
    return jnp.asarray(is_obs), jnp.asarray(obs_idx), n_obs
