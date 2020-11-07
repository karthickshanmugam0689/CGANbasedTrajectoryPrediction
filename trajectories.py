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
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, loss_mask_list, obs_ped_abs_speed,
     pred_ped_abs_speed, obs_label, pred_label) = zip(*data)

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
    #ped_features = torch.cat(ped_features, dim=0)
    obs_label = torch.cat(obs_label, dim=0).permute(2, 0, 1)
    pred_label = torch.cat(pred_label, dim=0).permute(2, 0, 1)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, obs_ped_abs_speed,
        pred_ped_abs_speed, obs_label, pred_label
    ]

    return tuple(out)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def read_file(_path, delim='\t'):
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            line = [float(i) for i in line]
            data.append(line[:5])
    return np.asarray(data)


def get_min_max_distance(seq_len, all_files):
    #ped_speed = []
    #small_vehicle_speed = []
    #big_vehicle_speed = []
    #cyclist_speed = []
    #other_speed = []
    all_speed, cyclist_speed, ped_speed, vehicle_speed, other_speed = [], [], [], [], []
    for path in all_files:
        data = read_file(path, ' ')
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :5])
        num_sequences = int(math.ceil((len(frames) - seq_len + 1)))

        for idx in range(0, num_sequences):
            curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            obj_in_curr_seq = np.unique(curr_seq_data[:, 1])
            for _, obj_id in enumerate(obj_in_curr_seq):
                curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]
                label = curr_obj_seq[0, 2]
                pad_front = frames.index(curr_obj_seq[0, 0]) - idx
                pad_end = frames.index(curr_obj_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != seq_len and label != 5:
                    continue
                curr_obj_x_axis = [np.square(t - s) for s, t in
                                           zip(curr_obj_seq[:, 3], curr_obj_seq[1:, 3])]
                curr_obj_y_axis = [np.square(t - s) for s, t in
                                           zip(curr_obj_seq[:, 4], curr_obj_seq[1:, 4])]
                curr_obj_dist = np.sqrt(np.add(curr_obj_x_axis, curr_obj_y_axis))
                curr_obj_speed = curr_obj_dist / 0.5
                all_speed.append(curr_obj_speed)
                if label == 4:
                    cyclist_speed.append(np.max(curr_obj_speed))
                    cyclist_speed.append(np.min(curr_obj_speed))
                elif label == 3:
                    ped_speed.append(np.max(curr_obj_speed))
                    ped_speed.append(np.min(curr_obj_speed))
                elif label == 1 or label == 2:
                    vehicle_speed.append(np.max(curr_obj_speed))
                    vehicle_speed.append(np.min(curr_obj_speed))
                #elif label == 2:
                #    big_vehicle_speed.append(np.max(curr_obj_speed))
                #    big_vehicle_speed.append(np.min(curr_obj_speed))

    all_speed = np.concatenate(all_speed, axis=0)
    max_speed = np.amax(all_speed)
    min_speed = np.min(all_speed)

    max_ped_speed = np.amax(ped_speed)
    min_ped_speed = np.min(ped_speed)
    max_veh_speed = np.amax(vehicle_speed)
    min_veh_speed = np.min(vehicle_speed)
    max_cyc_speed = np.amax(cyclist_speed)
    min_cyc_speed = np.min(cyclist_speed)
    #ped_speed = np.array(ped_speed).reshape(-1, 1)
    #small_vehicle_speed = np.array(small_vehicle_speed).reshape(-1, 1)
    #big_vehicle_speed = np.array(big_vehicle_speed).reshape(-1, 1)
    #cyclist_speed = np.array(cyclist_speed).reshape(-1, 1)
    #other_speed = np.array(other_speed).reshape(-1, 1)
#    fig, ax = plt.subplots(figsize=(16, 8))
#    ax.scatter(other_speed, other_speed)
#    plt.show()

    # Find the domain-wise max and min speed to normalize the speed values
    #if ped_speed.size != 0:
    #    max_ped_speed = np.amax(ped_speed)
    #    min_ped_speed = np.min(ped_speed)
    #else:
    #    max_ped_speed, min_ped_speed = 0, 0
    #if big_vehicle_speed.size != 0:
    #    max_big_vehicle_speed = np.amax(big_vehicle_speed)
    #    min_big_vehicle_speed = np.min(big_vehicle_speed)
    #else:
    #    max_big_vehicle_speed, min_big_vehicle_speed = 0, 0
    #if small_vehicle_speed.size != 0:
    #    max_small_vehicle_speed = np.amax(small_vehicle_speed)
    #    min_small_vehicle_speed = np.min(small_vehicle_speed)
    #else:
    #    max_small_vehicle_speed, min_small_vehicle_speed = 0, 0
    #if cyclist_speed.size != 0:
    #    max_cyclist_speed = np.amax(cyclist_speed)
    #    min_cyclist_speed = np.min(cyclist_speed)
    #else:
    #    max_cyclist_speed, min_cyclist_speed = 0, 0
    #if other_speed.size != 0:
    #    max_other_speed = np.amax(other_speed)
    #    min_other_speed = np.min(other_speed)
    #else:
    #    max_other_speed, min_other_speed = 0, 0

    return max_speed, min_speed


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, metric=0
    ):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        SEQ_LEN = OBS_LEN + PRED_LEN
        self.train_or_test = metric

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        obj_abs_speed = []
        obj_label = []
        features = []
        loss_mask_list = []
        for path in all_files:
            data = read_file(path, '\t')
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - SEQ_LEN + 1)))
            max_speed, min_speed = get_min_max_distance(SEQ_LEN, all_files)
            #if self.train_or_test == 1:
            #    min, max = get_min_max_speed_labels(num_sequences, frame_data, self.seq_len, frames)
            for idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + SEQ_LEN], axis=0)

                obj_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_loss_mask = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                curr_seq_rel = np.zeros((len(obj_in_curr_seq), 2, SEQ_LEN))
                curr_seq = np.zeros((len(obj_in_curr_seq), 2, SEQ_LEN))
                _curr_abs_speed = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                _curr_obj_label = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                num_peds_considered = 0

                for _, ped_id in enumerate(obj_in_curr_seq):
                    curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_obj_seq = np.around(curr_obj_seq, decimals=4)
                    pad_front = frames.index(curr_obj_seq[0, 0]) - idx
                    pad_end = frames.index(curr_obj_seq[-1, 0]) - idx + 1
                    label = curr_obj_seq[0, 2]
                    curr_seq_transpose = np.transpose(curr_obj_seq[:, 3:5])
                    if pad_end - pad_front == SEQ_LEN and curr_seq_transpose.shape[1] == SEQ_LEN and label != 5:
                        curr_ped_x_axis_new = [0.0] + [np.square(t - s) for s, t in
                                                       zip(curr_obj_seq[:, 3], curr_obj_seq[1:, 3])]
                        curr_ped_y_axis_new = [0.0] + [np.square(t - s) for s, t in
                                                       zip(curr_obj_seq[:, 4], curr_obj_seq[1:, 4])]

                        curr_dist = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
                        # Since each frame is taken with an interval of 0.4, we divide the distance with 0.4 to get speed
                        curr_abs_speed = curr_dist / 0.5
                        #curr_abs_speed = [(x - min_speed) / (max_speed - min_speed) if x > 0 else 0 for x in curr_abs_speed]
                        if label == 1 or label == 2:  # Small Vehicles and Big Vehicles considered as Vehcile
                            embedding_label = 0.1
                        elif label == 3:  # Pedestrians
                            embedding_label = 0.3
                        elif label == 4:  # Cyclist
                            embedding_label = 0.4
                        curr_abs_speed = np.around(curr_abs_speed, decimals=4)
                        curr_abs_speed = [sigmoid(x/10) for x in curr_abs_speed]
                        curr_abs_speed = np.around(curr_abs_speed, decimals=4)
                        curr_abs_speed = np.transpose(curr_abs_speed)

                        curr_obj_seq = np.transpose(curr_obj_seq[:, 3:5])
                        # Make coordinates relative
                        rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                        rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_obj_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_obj_seq
                        # Linear vs Non-Linear Trajectory
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        _curr_abs_speed[_idx, pad_front:pad_end] = curr_abs_speed
                        _curr_obj_label[_idx, pad_front:pad_end] = embedding_label
                        num_peds_considered += 1

                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    obj_abs_speed.append(_curr_abs_speed[:num_peds_considered])
                    obj_label.append(_curr_obj_label[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    #ped_seq = curr_seq[:num_peds_considered]
                    #ped_speed_feature = _curr_abs_speed[:num_peds_considered]

                    # DISTANCE, POSITION, SPEED FEATURE CONCAT EXTRACTION
                    # Calculating the nearby pedestrian distance and speed as a preprocessing step to increase the speed
                    # of model run
                    #max_ped_feature = np.zeros((num_peds_considered, 57, 3))
                    #last_pos_info = ped_seq[:, :, OBS_LEN - 1]
                    #next_pos_speed = ped_speed_feature[:, OBS_LEN]
                    #ped_wise_feature = []
                    #for a in last_pos_info:
                    #    curr_ped_feature = []
                    #    for b, speed in zip(last_pos_info, next_pos_speed):
                    #        if np.array_equal(a, b):
                    #            relative_pos = np.array([0.0, 0.0])
                    #        else:
                    #            relative_pos = b - a
                    #        speed = speed
                    #        concat = np.concatenate([relative_pos.reshape(1, 2), speed.reshape(1, 1)], axis=1)
                    #        curr_ped_feature.append(concat)
                    #    curr_ped_feature = np.concatenate(curr_ped_feature, axis=0)
                    #    ped_wise_feature.append(np.expand_dims(curr_ped_feature, axis=0))
                    #ped_wise_feature = np.concatenate(ped_wise_feature, axis=0)
                    #max_ped_feature[0:num_peds_considered, 0:num_peds_considered, :] = \
                    #    ped_wise_feature[0:num_peds_considered, :, :]
                    #features.append(max_ped_feature)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        obj_abs_speed = np.concatenate(obj_abs_speed, axis=0)
        obj_label = np.concatenate(obj_label, axis=0)
        #features = np.concatenate(features, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        obj_abs_speed = torch.from_numpy(obj_abs_speed).type(torch.float)
        #self.ped_features = torch.from_numpy(features).type(torch.float)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, OBS_LEN:]).type(torch.float)

        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, OBS_LEN:]).type(torch.float)

        self.obs_obj_abs_speed = obj_abs_speed[:, :OBS_LEN].unsqueeze(dim=1).type(torch.float)
        self.pred_obj_abs_speed = obj_abs_speed[:, OBS_LEN:].unsqueeze(dim=1).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)

        self.obs_obj_label = torch.from_numpy(obj_label[:, :OBS_LEN]).unsqueeze(dim=1).type(torch.float)
        self.pred_obj_label = torch.from_numpy(obj_label[:, OBS_LEN:]).unsqueeze(dim=1).type(torch.float)

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
            self.loss_mask[start:end, :], self.obs_obj_abs_speed[start:end, :],
            self.pred_obj_abs_speed[start:end, :], self.obs_obj_label[start:end, :],
            self.pred_obj_label[start:end, :]
        ]
        return out