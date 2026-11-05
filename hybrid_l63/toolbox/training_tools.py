import grain.python as grain
import numpy as np
from typing import Dict


class TimeSeriesDataSource(grain.RandomAccessDataSource):
    """Maps window-start states to full state windows.

    X = state at the window start.
    Y = the full window [start .. start + n_rollsteps] (Y[0] == X), so both
    'terminal' (Y[-1]) and 'running' (Y[1:]) loss_eval modes are supported
    downstream without touching the dataloader again.
    """

    def __init__(self, data: np.ndarray, n_rollsteps: int, stride: int = 1):
        self._data = data
        self._n_rollsteps = n_rollsteps
        self._stride = stride
        span = n_rollsteps + 1
        self._num_sequences = max(0, (len(data) - span) // stride + 1)

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        start = index * self._stride
        window = self._data[start: start + self._n_rollsteps + 1]

        return {
            'X': window[0].astype(np.float32),
            'Y': window.astype(np.float32),
        }


def set_dataloaders(
    ref_trajectory,
    config_training,
    train_ratio: float = 0.7,
    discard_ratio: float = 0.1,
    test_ratio: float = 0.2,
    shuffle: bool = True,
    train_stride: int = 1,
):
    """
    Split the reference trajectory into contiguous train/discard/test time blocks,
    then build a dataloader per block.

    Train windows overlap (stride=train_stride, default 1) to get many training
    sequences out of a single chaotic trajectory. Discard/test windows are
    non-overlapping (stride = n_rollsteps + 1) so evaluation isn't inflated by
    near-duplicate, highly autocorrelated windows. The discard block is a gap
    carved out between train and test and is not turned into a loader.

    Returns:
        Dict with 'train', 'test' dataloaders.
    """
    assert abs(train_ratio + discard_ratio + test_ratio - 1.0) < 1e-8, \
        "train_ratio + discard_ratio + test_ratio must sum to 1"

    data = np.asarray(ref_trajectory)
    n_rollsteps = config_training.n_rollsteps
    span = n_rollsteps + 1

    total_timesteps = len(data)
    train_end = int(total_timesteps * train_ratio)
    discard_end = train_end + int(total_timesteps * discard_ratio)

    train_block = data[:train_end]
    discard_block = data[train_end:discard_end]
    test_block = data[discard_end:]

    train_source = TimeSeriesDataSource(train_block, n_rollsteps, stride=train_stride)
    test_source = TimeSeriesDataSource(test_block, n_rollsteps, stride=span)

    print(f"Total timesteps: {total_timesteps}")
    print(f"  Train:   {len(train_block)} steps -> {len(train_source)} sequences "
          f"(stride={train_stride}, overlapping)")
    print(f"  Discard: {len(discard_block)} steps (gap, unused)")
    print(f"  Test:    {len(test_block)} steps -> {len(test_source)} sequences "
          f"(stride={span}, non-overlapping)")

    seed = 42
    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=grain.IndexSampler(
            num_records=len(train_source),
            shuffle=shuffle,
            seed=seed,
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=config_training.batch_size, drop_remainder=True)],
    )

    test_loader = grain.DataLoader(
        data_source=test_source,
        sampler=grain.IndexSampler(
            num_records=len(test_source),
            shuffle=False,
            num_epochs=1,
        ),
        operations=[grain.Batch(batch_size=config_training.batch_size, drop_remainder=False)],
    )

    return {
        'train': train_loader,
        'test': test_loader,
    }
