import numpy as np
import jax.numpy as jnp

from typing import NamedTuple


class GradApproxConfig(NamedTuple):
    anchor_pos:   jnp.ndarray   # (n_chunks,)
    step_indices: jnp.ndarray   # (n_chunks, max_k)
    n_chunks:     int
    max_k:        int
    pad_len:      int
    nsteps:       int

def make_chunk_anchor_indices(
    nsteps:    int,
    n_anchors: int,
    bias_dir:  str  = "upstream",
    verbose:   bool = False,
) -> tuple[jnp.ndarray, np.ndarray, int]:
    """
    Divide nsteps into exactly n_anchors chunks.
    All chunks are base_k = nsteps // n_anchors steps, except at most ONE
    chunk which absorbs the entire remainder (nsteps % n_anchors) -- kept as
    a single uneven chunk rather than spread one-extra-step-each across
    `remainder` chunks, so there's never more than one non-uniform chunk to
    reason about. bias_dir controls which end that one chunk sits at:
        upstream   → the larger chunk is first  (forward time)
        downstream → the larger chunk is last   (forward time)

    Anchor always at middle of each chunk's REAL steps.

    Returns:
        anchor_pos  : (n_anchors,) int — local index within each chunk
        chunk_sizes : (n_anchors,) int — size of each chunk (including pad)
        pad_len     : int — total padding added
    """
    assert bias_dir in ("upstream", "downstream")
    assert n_anchors <= nsteps, "n_anchors must be <= nsteps"

    base_k    = nsteps // n_anchors
    remainder = nsteps %  n_anchors   # absorbed by a single chunk, not spread

    chunk_sizes = np.full(n_anchors, base_k, dtype=np.int32)
    if remainder > 0:
        if bias_dir == "upstream":
            chunk_sizes[0] += remainder     # single larger chunk, at the start
        else:
            chunk_sizes[-1] += remainder    # single larger chunk, at the end

    # pad so all chunks are same size (max chunk size)
    max_k   = int(chunk_sizes.max())
    pad_per = max_k - chunk_sizes            # padding per chunk
    pad_len = int(pad_per.sum())

    # anchor position within each chunk (middle of real steps)
    anchor_pos = np.zeros(n_anchors, dtype=np.int32)
    for i in range(n_anchors):
        real = chunk_sizes[i]
        if real % 2 == 1:
            anchor_pos[i] = real // 2
        else:
            anchor_pos[i] = real // 2 - 1 if bias_dir == "upstream" else real // 2

    # chunk starts in global step indexing
    chunk_starts = np.concatenate([[0], np.cumsum(chunk_sizes[:-1])])

    if verbose:
        print(f"\n── Chunk layout ─────────────────────────────")
        print(f"  nsteps={nsteps}, n_anchors={n_anchors} (exact)")
        print(f"  base_k={base_k}, remainder={remainder}")
        print(f"  max_k={max_k} (padded chunk size), total_pad={pad_len}")
        print(f"  bias_dir={bias_dir!r}")
        print(f"─────────────────────────────────────────────")
        for i in range(n_anchors):
            start        = chunk_starts[i]
            real         = chunk_sizes[i]
            pad          = pad_per[i]
            anc_local    = anchor_pos[i]
            anc_global   = start + anc_local
            steps_str    = f"[{start}..{start+real-1}]"
            pad_str      = f" + {pad} padded" if pad > 0 else ""
            print(f"  chunk {i:2d}: steps {steps_str:14s} "
                  f"real={real}{pad_str:12s} "
                  f"anchor_local={anc_local}  anchor_global={anc_global}")
        print(f"─────────────────────────────────────────────\n")

    return jnp.array(anchor_pos), chunk_sizes, max_k


def setup_grad_approx(
    nsteps:    int,
    n_anchors: int,
    bias_dir:  str  = "upstream",
    verbose:   bool = False,
) -> GradApproxConfig:
    """Pure numpy — run once outside everything."""
    anchor_pos, chunk_sizes, max_k = make_chunk_anchor_indices(
        nsteps, n_anchors, bias_dir, verbose=verbose
    )
    n_chunks = len(chunk_sizes)
    pad_len  = int(n_chunks * max_k - nsteps)
    chunk_starts = np.concatenate([[0], np.cumsum(np.asarray(chunk_sizes)[:-1])])
    step_indices_np = np.full((n_chunks, max_k), nsteps, dtype=np.int32)
    for i in range(n_chunks):
        start, size = int(chunk_starts[i]), int(chunk_sizes[i])
        step_indices_np[i, :size] = np.arange(start, start + size)
    step_indices = jnp.array(step_indices_np)

    return GradApproxConfig(
        anchor_pos   = anchor_pos,
        step_indices = step_indices,
        n_chunks     = n_chunks,
        max_k        = max_k,
        pad_len      = pad_len,
        nsteps       = nsteps,
    )


def _anchor_global_indices(config: GradApproxConfig):
    """Resolves the precise global timeline index for each chunk's designated anchor step."""
    chunk_range = jnp.arange(config.n_chunks)
    return config.step_indices[chunk_range, config.anchor_pos]


def report_tbptt_chunk_coverage(config: GradApproxConfig, obs_stride: int) -> list[int]:
    """Prints a chunk-by-chunk recap (size, real steps, observation count) and
    flags any chunk with zero observations.
    """
    from toolbox.loss_tools import build_obs_index
    is_obs, _, _ = build_obs_index(config.nsteps, obs_stride)
    step_indices = np.asarray(config.step_indices)

    print(f"TBPTT chunk layout: {config.n_chunks} chunks, nsteps={config.nsteps}, obs_stride={obs_stride}")
    empty_chunks = []
    for i, row in enumerate(step_indices):
        real_steps = [int(s) for s in row if s < config.nsteps]
        n_obs_in_chunk = sum(1 for s in real_steps if is_obs[s])
        print(f"  chunk {i}: size={len(real_steps)}  steps={real_steps}  n_obs={n_obs_in_chunk}")
        if n_obs_in_chunk == 0:
            print(f"    -> no observation in chunk {i}: zero gradient injection for this chunk")
            empty_chunks.append(i)

    return empty_chunks
