from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import flax
import diffrax


_SOLVERS = {
    "Tsit5":  diffrax.Tsit5,
    "Euler":  diffrax.Euler,
    "Dopri5": diffrax.Dopri5,
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModelConfig:
    solver: str   = "Tsit5"
    dt0:    float = 0.01
    hybrid: bool  = False
    dtype:  Any   = jnp.float32

    def make_solver(self):
        try:
            return _SOLVERS[self.solver]()
        except KeyError:
            raise ValueError(f"Unknown solver {self.solver!r}. Choose from {list(_SOLVERS)}")


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class ModelState(NamedTuple):
    x: jnp.ndarray              # (3,)  [x, y, z]
    t: jnp.ndarray | None = None  # ()  optional time tracking

    @classmethod
    def init(
        cls,
        x:   float = 8.0,
        y:   float = 0.0,
        z:   float = 30.0,
        t:   float | None = None,
        dtype: Any = jnp.float32,
    ) -> ModelState:
        return cls(
            x=jnp.array([x, y, z], dtype=dtype),
            t=jnp.array(t, dtype=dtype) if t is not None else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PARAMS
# ══════════════════════════════════════════════════════════════════════════════
class ModelParams(NamedTuple):
    sigma: float = 10.0     
    rho:   float = 28.0
    beta:  float = 8.0 / 3.0
    
class SubmodelParams(NamedTuple):
    theta: dict = None

class Params(NamedTuple):
    model:    ModelParams
    submodel: SubmodelParams