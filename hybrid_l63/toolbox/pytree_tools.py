import jax
import jax.numpy as jnp
import equinox as eqx


def _zeros(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)

def _add(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)

def params_partition(params, mask):
    """Splits params into dynamic and static parts without dropping types."""
    dynamic = eqx.filter(params, mask)
    static = eqx.filter(params, jax.tree_util.tree_map(lambda m: not m, mask))
    return dynamic, static


def params_recombine(dynamic, static):
    """Recombines the structures back together, preserving types for JAX."""
    return eqx.combine(dynamic, static)
