import logging
import os
import math

import numpy as np
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
from constants import *

logger = logging.getLogger(__name__)


def data_loader(path, metric):
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


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, loss_mask_list, obs_ped_abs_speed, pred_ped_abs_speed,
     ped_features) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_ped_abs_speed = torch.cat(obs_ped_abs_speed, dim=0).permute(2, 0, 1)
    pred_ped_abs_speed = torch.cat(pred_ped_abs_speed, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    ped_features = torch.cat(ped_features, dim=0)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, obs_ped_abs_speed,
        pred_ped_abs_speed, ped_features
    ]

    return tuple(out)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def read_file(_path, delim='\t'):
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def get_min_max_speed_labels(num_sequences, frame_data, seq_len, frames):
    ped_speed = []
    for idx in range(0, num_sequences):
        curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
        ped_in_curr_seq = np.unique(curr_seq_data[:, 1])
        for _, obj_id in enumerate(ped_in_curr_seq):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
            if pad_end - pad_front != seq_len:
                continue
            curr_ped_x_axis_new = [0.0] + [np.square(t - s) for s, t in
                                           zip(curr_ped_seq[:, 2], curr_ped_seq[1:, 2])]
            curr_ped_y_axis_new = [0.0] + [np.square(t - s) for s, t in
                                           zip(curr_ped_seq[:, 3], curr_ped_seq[1:, 3])]

            curr_ped_dist = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
            curr_ped_abs_speed = curr_ped_dist / 0.4
            ped_speed.append(curr_ped_abs_speed)
    ped_speed = np.array(ped_speed).reshape(-1, 1)
    max_ped_speed = np.amax(ped_speed)
    min_ped_speed = np.min(ped_speed)
    return max_ped_speed, min_ped_speed


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
            self, data_dir, metric=0
    ):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.seq_len = OBS_LEN + PRED_LEN
        self.train_or_test = metric

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        ped_abs_speed = []
        features = []
        loss_mask_list = []
        for path in all_files:
            data = read_file(path, '\t')
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1)))
            # Uncomment the below lines to test the max and min speeds available in the test datasets.
            # This value is multiplied with the user speed from 0 to 1 - thus reflecting 1 as max speed and 0 as min speed
            #if self.train_or_test == 1:
            #    min, max = get_min_max_speed_labels(num_sequences, frame_data, self.seq_len, frames)
            for idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                _curr_ped_abs_speed = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    # for manhattan, change np.square to np.abs
                    curr_ped_x_axis_new = [0.0] + [np.square(t - s) for s, t in
                                           zip(curr_ped_seq[:, 2], curr_ped_seq[1:, 2])]
                    curr_ped_y_axis_new = [0.0] + [np.square(t - s) for s, t in
                                           zip(curr_ped_seq[:, 3], curr_ped_seq[1:, 3])]

                    curr_ped_dist = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
                    # Since each frame is taken with an interval of 0.4, we divide the distance with 0.4 to get speed
                    curr_ped_abs_speed = curr_ped_dist / 0.4
                    curr_ped_abs_speed = [sigmoid(x) for x in curr_ped_abs_speed]
                    curr_ped_abs_speed = np.around(curr_ped_abs_speed, decimals=4)
                    curr_ped_abs_speed = np.transpose(curr_ped_abs_speed)

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    _curr_ped_abs_speed[_idx, pad_front:pad_end] = curr_ped_abs_speed
                    num_peds_considered += 1

                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    ped_abs_speed.append(_curr_ped_abs_speed[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    ped_seq = curr_seq[:num_peds_considered]
                    ped_speed_feature = _curr_ped_abs_speed[:num_peds_considered]

                    # DISTANCE, POSITION, SPEED FEATURE CONCAT EXTRACTION
                    # Calculating the nearby pedestrian distance and speed as a preprocessing step to increase the speed
                    # of model run
                    max_ped_feature = np.zeros((num_peds_considered, 57, 3))
                    last_pos_info = ped_seq[:, :, self.obs_len - 1]
                    next_pos_speed = ped_speed_feature[:, self.obs_len]
                    ped_wise_feature = []
                    for a in last_pos_info:
                        curr_ped_feature = []
                        for b, speed in zip(last_pos_info, next_pos_speed):
                            if np.array_equal(a, b):
                                relative_pos = np.array([0.0, 0.0])
                            else:
                                relative_pos = b - a
                            speed = speed
                            concat = np.concatenate([relative_pos.reshape(1, 2), speed.reshape(1, 1)], axis=1)
                            curr_ped_feature.append(concat)
                        curr_ped_feature = np.concatenate(curr_ped_feature, axis=0)
                        ped_wise_feature.append(np.expand_dims(curr_ped_feature, axis=0))
                    ped_wise_feature = np.concatenate(ped_wise_feature, axis=0)
                    max_ped_feature[0:num_peds_considered, 0:num_peds_considered, :] = \
                        ped_wise_feature[0:num_peds_considered, :, :]
                    features.append(max_ped_feature)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        ped_abs_speed = np.concatenate(ped_abs_speed, axis=0)
        features = np.concatenate(features, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        ped_abs_speed = torch.from_numpy(ped_abs_speed).type(torch.float)
        self.ped_features = torch.from_numpy(features).type(torch.float)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, OBS_LEN:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, OBS_LEN:]).type(torch.float)
        self.obs_ped_abs_speed = ped_abs_speed[:, :OBS_LEN]
        self.obs_ped_abs_speed = self.obs_ped_abs_speed.unsqueeze(dim=1)
        self.pred_ped_abs_speed = ped_abs_speed[:, OBS_LEN:]
        self.pred_ped_abs_speed = self.pred_ped_abs_speed.unsqueeze(dim=1)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.loss_mask[start:end, :], self.obs_ped_abs_speed[start:end, :],
            self.pred_ped_abs_speed[start:end, :], self.ped_features[start:end, :]
        ]
        return out