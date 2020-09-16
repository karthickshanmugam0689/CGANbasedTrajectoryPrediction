import pickle

import torch

from csgan.data.trajectories import data_loader
from csgan.models import TrajectoryGenerator
from csgan.losses import displacement_error, final_displacement_error
from csgan.losses import relative_to_abs
from csgan.constants import *


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    simulated_output = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
            ped_features) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel, logged_output = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                              TEST_METRIC, SPEED_TO_ADD, ped_features)

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

                simulated_output.append(logged_output)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * PRED_LEN)
        fde = sum(fde_outer) / (total_traj)
        with open('ResultTrajectories.pkl', 'wb') as f:
            pickle.dump(simulated_output, f, pickle.HIGHEST_PROTOCOL)
        return ade, fde


def main():
    checkpoint = torch.load(CHECKPOINT_NAME)
    generator = TrajectoryGenerator()
    generator.load_state_dict(checkpoint['g_state'])
    generator.train()

    path = TEST_DATASET_PATH
    _, loader = data_loader(path, TEST_METRIC)
    ade, fde = evaluate(loader, generator, NUM_SAMPLES)
    print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()
