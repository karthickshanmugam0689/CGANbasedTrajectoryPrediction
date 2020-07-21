import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, obs_ped_abs_speed, pred_ped_abs_speed) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    obs_ped_abs_speed = torch.cat(obs_ped_abs_speed, dim=0).permute(2, 0, 1)
    pred_ped_abs_speed = torch.cat(pred_ped_abs_speed, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, obs_ped_abs_speed, pred_ped_abs_speed
    ]

    return tuple(out)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    if delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def get_min_max_speed_labels(num_sequences, frame_data, seq_len, frames):
    ped_speed = []
    for idx in range(0, num_sequences):
        curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
        for _, ped_id in enumerate(peds_in_curr_seq):
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
            if pad_end - pad_front != seq_len:
                continue
            curr_ped_x_axis_new = [0.0] + [np.abs(s - t) for s, t in
                                           zip(curr_ped_seq[:, 2], curr_ped_seq[1:, 2])]
            curr_ped_y_axis_new = [0.0] + [np.abs(s - t) for s, t in
                                           zip(curr_ped_seq[:, 3], curr_ped_seq[1:, 3])]
            curr_ped_dist_formula = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
            curr_ped_speed = curr_ped_dist_formula / 0.4
            ped_speed.append(curr_ped_speed)
    ped_speed = np.asarray(ped_speed).reshape(-1, 1)
    max_speed = np.around(np.amax(ped_speed), decimals=4)
    min_speed = np.around(np.min(ped_speed), decimals=4)
    return min_speed, max_speed

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        ped_abs_speed = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            min_speed, max_speed = get_min_max_speed_labels(num_sequences, frame_data, self.seq_len, frames)
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                curr_ped_frame_seq = np.unique(curr_seq_data[:, 0])
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                _curr_ped_abs_speed = np.zeros((len(peds_in_curr_seq), self.seq_len))
                _curr_ped_rel_speed = np.zeros((len(peds_in_curr_seq), self.seq_len))
                _curr_ped_dist = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
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
                    curr_ped_abs_speed_a = curr_ped_dist / 0.4
                    curr_ped_abs_speed = [(x - min_speed)/(max_speed - min_speed) for x in curr_ped_abs_speed_a]
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
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    _curr_ped_dist[_idx, pad_front:pad_end] = curr_ped_dist
                    _curr_ped_abs_speed[_idx, pad_front:pad_end] = curr_ped_abs_speed
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    ped_abs_speed.append(_curr_ped_abs_speed[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        ped_abs_speed = np.concatenate(ped_abs_speed, axis=0)
        ped_abs_speed = torch.from_numpy(ped_abs_speed).type(torch.float)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.obs_ped_abs_speed = ped_abs_speed[:, :self.obs_len]
        self.obs_ped_abs_speed = self.obs_ped_abs_speed.unsqueeze(dim=1)
        self.pred_ped_abs_speed = ped_abs_speed[:, self.obs_len:]
        self.pred_ped_abs_speed = self.pred_ped_abs_speed.unsqueeze(dim=1)
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
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.obs_ped_abs_speed[start:end, :], self.pred_ped_abs_speed[start:end, :]
        ]
        return out
