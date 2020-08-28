from torch.utils.data import DataLoader
from csgan.constants import *
from csgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path, metric):
    dset = TrajectoryDataset(
        path,
        metric)

    loader = DataLoader(
        dset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=seq_collate)
    return dset, loader
