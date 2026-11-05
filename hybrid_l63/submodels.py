import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any 

class SimpleFC(nn.Module):
    input_dim: int
    hidden_dim: int 
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
    
    def init_theta(self, seed: int = 0, dtype: Any = jnp.float32) -> dict:
        """
        Initialise and return theta as a plain dict pytree.

        * For ensemble initialization:
            keys   = jax.random.split(jax.random.PRNGKey(0), N)
            thetas = jax.vmap(lambda k: submodel.init_theta(cfg, seed=int(k)))(keys)
        """
        return self.init(
            jax.random.PRNGKey(seed),
            jnp.ones(3, dtype=dtype),
        )