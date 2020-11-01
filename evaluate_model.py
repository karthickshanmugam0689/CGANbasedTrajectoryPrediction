import pickle
import torch
import os

from train import get_diff_traj
from trajectories import data_loader
from models import TrajectoryGenerator
from utils import displacement_error, final_displacement_error, relative_to_abs
from constants import *


def evaluate_helper(error, seq_start_end):
    sum_ = []
    for (start, end) in seq_start_end:
        _error = sum_.append(torch.min(torch.sum(error[start.item():end.item()], dim=0)))
    return sum(sum_)


def evaluate(loader, generator, num_samples):
    ade_outer, fde_outer, simulated_output, total_traj, veh_traj, ped_traj, cyc_traj = [], [], [], [], [], [], []
    veh_disp_error_outer, ped_disp_error_outer, cyc_disp_error_outer = [], [], []
    veh_f_disp_error_outer, ped_f_disp_error_outer, cyc_f_disp_error_outer = [], [], []
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_label, pred_label) = batch

            ade, veh_disp_error, ped_disp_error, cyc_disp_error = [], [], [], []
            fde, veh_f_disp_error, ped_f_disp_error, cyc_f_disp_error = [], [], [], []
            total_traj.append(pred_traj_gt.size(1))
            _, _veh_gt, _, _ped_gt, _, _cyc_gt = get_diff_traj(pred_traj_gt, pred_traj_gt, pred_label)
            veh_traj.append(_veh_gt.size(1))
            ped_traj.append(_ped_gt.size(1))
            cyc_traj.append(_cyc_gt.size(1))

            for _ in range(num_samples):
                if TEST_METRIC == 1:
                    pred_traj_fake_rel, logged_output = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                              TEST_METRIC, SPEED_TO_ADD, obs_label, pred_label)
                else:
                    pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                pred_ped_speed, pred_traj_gt, TEST_METRIC, SPEED_TO_ADD, obs_label, pred_label)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                veh_fake, veh_gt, ped_fake, ped_gt, cyc_fake, cyc_gt = get_diff_traj(pred_traj_gt, pred_traj_fake, pred_label)
                _, veh_count, _ = veh_fake.size()
                _, ped_count, _ = ped_fake.size()
                _, cyc_count, _ = cyc_fake.size()

                veh_disp, _ = displacement_error(veh_fake, veh_gt, mode='raw')
                veh_disp_error.append(veh_disp)

                ped_disp, _ = displacement_error(ped_fake, ped_gt, mode='raw')
                ped_disp_error.append(ped_disp)

                cyc_disp, _ = displacement_error(cyc_fake, cyc_gt, mode='raw')
                cyc_disp_error.append(cyc_disp)

                overall_disp, _ = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw')
                ade.append(overall_disp)

                veh_f_disp_error.append(final_displacement_error(veh_fake[-1], veh_gt[-1], mode='raw'))
                ped_f_disp_error.append(final_displacement_error(ped_fake[-1], ped_gt[-1], mode='raw'))
                cyc_f_disp_error.append(final_displacement_error(cyc_fake[-1], cyc_gt[-1], mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                if TEST_METRIC:
                    simulated_output.append(logged_output)

            ade_outer.append(evaluate_helper(torch.stack(ade, dim=1), seq_start_end))
            veh_disp_error_outer.append(evaluate_helper(torch.stack(veh_disp_error, dim=1), seq_start_end))
            ped_disp_error_outer.append(evaluate_helper(torch.stack(ped_disp_error, dim=1), seq_start_end))
            cyc_disp_error_outer.append(evaluate_helper(torch.stack(cyc_disp_error, dim=1), seq_start_end))

            fde_outer.append(evaluate_helper(torch.stack(fde, dim=1), seq_start_end))
            veh_f_disp_error_outer.append(evaluate_helper(torch.stack(veh_f_disp_error, dim=1), seq_start_end))
            ped_f_disp_error_outer.append(evaluate_helper(torch.stack(ped_f_disp_error, dim=1), seq_start_end))
            cyc_f_disp_error_outer.append(evaluate_helper(torch.stack(cyc_f_disp_error, dim=1), seq_start_end))

        ade = sum(ade_outer) / (sum(total_traj) * PRED_LEN)
        fde = sum(fde_outer) / (sum(total_traj))

        veh_ade = sum(veh_disp_error_outer) / (sum(veh_traj) * PRED_LEN)
        veh_fde = sum(veh_f_disp_error_outer) / (sum(veh_traj))

        ped_ade = sum(ped_disp_error_outer) / (sum(ped_traj) * PRED_LEN)
        ped_fde = sum(ped_f_disp_error_outer) / (sum(ped_traj))

        cyc_ade = sum(cyc_disp_error_outer) / (sum(cyc_traj) * PRED_LEN)
        cyc_fde = sum(cyc_f_disp_error_outer) / (sum(cyc_traj))

        #if TEST_METRIC:
        #    with open('ResultTrajectories.pkl', 'wb') as f:
        #        pickle.dump(simulated_output, f, pickle.HIGHEST_PROTOCOL)
        return ade, fde, veh_ade, veh_fde, ped_ade, ped_fde, cyc_ade, cyc_fde


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
    ade, fde, veh_ade, veh_fde, ped_ade, ped_fde, cyc_ade, cyc_fde = evaluate(loader, generator, num_samples)
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(dataset_name, PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()
