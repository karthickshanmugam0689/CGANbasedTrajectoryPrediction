import pickle
import torch
import os

from trajectories import data_loader
from models import TrajectoryGenerator
from utils import displacement_error, final_displacement_error, relative_to_abs
from constants import *


def evaluate_helper(error, seq_start_end):
    sum_ = []
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        _error = sum_.append(torch.min(torch.sum(error[start.item():end.item()], dim=0)))
    return sum(sum_)


def evaluate(loader, generator, num_samples):
    ade_outer, fde_outer, simulated_output, total_traj = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
            ped_features) = batch

            ade, fde = [], []
            total_traj.append(pred_traj_gt.size(1))

            for _ in range(num_samples):
                if TEST_METRIC == 1:
                    pred_traj_fake_rel, logged_output = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                              TEST_METRIC, SPEED_TO_ADD, ped_features)
                else:
                    pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                pred_ped_speed, pred_traj_gt, TEST_METRIC, SPEED_TO_ADD, ped_features)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                if TEST_METRIC:
                    simulated_output.append(logged_output)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (sum(total_traj) * PRED_LEN)
        fde = sum(fde_outer) / (sum(total_traj))
        if TEST_METRIC:
            with open('ResultTrajectories.pkl', 'wb') as f:
                pickle.dump(simulated_output, f, pickle.HIGHEST_PROTOCOL)
        return ade, fde


def get_dataset_name(path):
    dataset_name = os.path.basename(os.path.dirname(path))
    return dataset_name


def main():
    checkpoint = torch.load(CHECKPOINT_NAME)
    generator = TrajectoryGenerator()
    generator.load_state_dict(checkpoint['g_state'])
    if USE_GPU:
        generator.cuda()
        generator.train()
    else:
        generator.train()

    dataset_name = get_dataset_name(TEST_DATASET_PATH)
    _, loader = data_loader(TEST_DATASET_PATH, TEST_METRIC)
    if TEST_METRIC == 1:
        num_samples = 1
    else:
        num_samples = NUM_SAMPLES
    ade, fde = evaluate(loader, generator, num_samples)
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(dataset_name, PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()
