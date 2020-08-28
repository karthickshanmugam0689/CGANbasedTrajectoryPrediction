import argparse
import os
import torch

from attrdict import AttrDict

from csgan.data.loader import data_loader
from csgan.models import TrajectoryGenerator
from csgan.losses import displacement_error, final_displacement_error
from csgan.utils import relative_to_abs, get_dset_path
from csgan.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--speed_to_add', default=0, type=float)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(obs_len=OBS_LEN, pred_len=PRED_LEN, embedding_dim=EMBEDDING_DIM,
                                    encoder_h_dim=ENCODER_H_DIM, decoder_h_dim=DECODER_H_DIM,
                                    mlp_dim=MLP_DIM, num_layers=NUM_LAYERS, noise_dim=NOISE_DIM,
                                    noise_type=NOISE_TYPE, noise_mix_type=NOISE_MIX_TYPE,
                                    dropout=DROPOUT, bottleneck_dim=BOTTLENECK_DIM,
                                    batch_norm=BATCH_NORM, embedding_dim_pooling=EMBEDDING_DIM)
    generator.load_state_dict(checkpoint['g_state'])
    generator.train()
    return generator


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


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            if USE_GPU == 0:
                batch = [tensor for tensor in batch]
            else:
                batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
             ped_features) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = \
                    generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                              args.train_or_test, SPEED_TO_ADD, ped_features)

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * PRED_LEN)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    test_metric = 1
    checkpoint = torch.load(CHECKPOINT_NAME)
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    path = TEST_DATASET_PATH
    _, loader = data_loader(_args, path, test_metric)
    ade, fde = evaluate(_args, loader, generator, NUM_SAMPLES)
    print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(PRED_LEN, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
