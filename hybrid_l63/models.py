import jax
import jax.numpy as jnp
import diffrax
from configs import ModelConfig, ModelState, Params

class L63:
    """Classic Lorenz 63 model"""

    def __init__(self, cfg: ModelConfig):
        self.cfg    = cfg
        # self.solver = cfg.make_solver()  # diffrax path kept below, commented out

    def rhs(self, state, params: Params, submodel=None)-> jnp.ndarray:
        x, y, z = state
        p = params.model
        xdot = p.sigma * (y - x)
        ydot = x * (p.rho - z) - y
        zdot = x * y - p.beta * z
        return jnp.array([xdot,ydot,zdot])

    def step(self, state: ModelState, params: Params, submodel=None) -> ModelState:
        x_next = rk4_step(lambda x: self.rhs(x, params), state.x, self.cfg.dt0)
        t_new = (state.t + self.cfg.dt0) if state.t is not None else None
        return ModelState(x=x_next, t=t_new)

    # def step(self, state: ModelState, params: Params, submodel=None) -> ModelState:
    #     term = diffrax.ODETerm(lambda t, y, a: self.rhs(y, t, a))
    #     sol  = diffrax.diffeqsolve(
    #         term, self.solver,
    #         t0=0.0, t1=self.cfg.dt0, dt0=self.cfg.dt0,
    #         y0=state.x, saveat=diffrax.SaveAt(t1=True),
    #         args=params,
    #         adjoint=diffrax.DirectAdjoint(),
    #
    #     )
    #     t_new = (state.t + self.cfg.dt0) if state.t is not None else None
    #     return ModelState(x=sol.ys[0], t=t_new)

# -------------------------------------------------------------------------------------------------------------------

class L63_incomplete:
    """Incomplete version of Lorenz 63 model where the third equation in the system is missing a term.
    Possible to branch a data-driven submodel and go hybrid
    """
    def __init__(self, cfg: ModelConfig):
        self.cfg    = cfg
        # self.solver = cfg.make_solver()  # diffrax path kept below, commented out

    def rhs(self, state, params: Params, submodel=None) -> jnp.ndarray:
        x, y, z = state
        p = params.model
        xdot = p.sigma * (y - x)
        ydot = x * (p.rho - z) - y
        zdot = x * y
        dt = jnp.array([xdot,ydot,zdot])
        if submodel is not None and params.submodel.theta is not None:
            dt = dt + submodel.apply(params.submodel.theta, state)
        return dt

    def step(self, state: ModelState, params: Params, submodel=None) -> ModelState:
        x_next = rk4_step(lambda x: self.rhs(x, params, submodel), state.x, self.cfg.dt0)
        t_new = (state.t + self.cfg.dt0) if state.t is not None else None
        return ModelState(x=x_next, t=t_new)

    # def step(self, state:   ModelState,params:  Params, submodel = None) -> ModelState:
    #     term = diffrax.ODETerm(lambda t, y, a: self.rhs(y, t, a, submodel))
    #     sol  = diffrax.diffeqsolve(
    #         term, self.solver,
    #         t0=0.0, t1=self.cfg.dt0, dt0=self.cfg.dt0,
    #         y0=state.x, saveat=diffrax.SaveAt(t1=True),
    #         args=params,
    #         adjoint=diffrax.DirectAdjoint(),
    #     )
    #     t_new = (state.t + self.cfg.dt0) if state.t is not None else None
    #     return ModelState(x=sol.ys[0], t=t_new)

def rk4_step(rhs_fn, x, dt):
    """Classic explicit RK4 step for an autonomous ODE dx/dt = rhs_fn(x).

    L63 is only ever advanced by exactly one fixed-size step (t1-t0 == dt0,
    ConstantStepSize), so there's nothing adaptive step-size control or
    adjoint-mode selection buys here -- a plain fixed-step RK4 gives the same
    per-step accuracy diffrax's Tsit5 did in this regime, with residuals that
    are just a few rhs_fn evaluations, so jax.vjp/memory_analysis profiling
    reflects the model itself rather than a solver's internal machinery.
    """
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)