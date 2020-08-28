import argparse
import gc
import logging
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from csgan.constants import *

from csgan.data.loader import data_loader
from csgan.losses import gan_g_loss, gan_d_loss, l2_loss
from csgan.losses import displacement_error, final_displacement_error

from csgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from csgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Generator Options
parser.add_argument('--noise_type', default='gaussian')

# Output
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--checkpoint_every', default=5, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--output_dir', default=os.getcwd())



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main(args):
    train_metric = 0
    print("Process Started")
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(args, TRAIN_DATASET_PATH, train_metric)
    print("Initializing val dataset")
    _, val_loader = data_loader(args, VAL_DATASET_PATH, train_metric)

    iterations_per_epoch = len(train_dset) / BATCH / D_STEPS
    if NUM_EPOCHS:
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator = TrajectoryGenerator(
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        embedding_dim=EMBEDDING_DIM,
        encoder_h_dim=ENCODER_H_DIM,
        decoder_h_dim=DECODER_H_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        noise_dim=NOISE_DIM,
        noise_type=NOISE_TYPE,
        noise_mix_type=NOISE_MIX_TYPE,
        dropout=DROPOUT,
        bottleneck_dim=BOTTLENECK_DIM,
        batch_norm=BATCH_NORM)

    generator.apply(init_weights)
    if USE_GPU == 0:
        generator.type(torch.FloatTensor).train()
    else:
        generator.type(torch.cuda.FloatTensor).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator()

    discriminator.apply(init_weights)
    if USE_GPU == 0:
        discriminator.type(torch.FloatTensor).train()
    else:
        discriminator.type(torch.cuda.FloatTensor).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < NUM_EPOCHS:
        gc.collect()
        d_steps_left, g_steps_left = D_STEPS, G_STEPS
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, limit=True)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = CHECKPOINT_NAME
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                break


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    if USE_GPU == 0:
        batch = [tensor for tensor in batch]
    else:
        batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
     ped_features) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                              pred_traj_gt, args.train_or_test, SPEED_TO_ADD, ped_features)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, ped_speed, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return losses


def generator_step(
        args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    if USE_GPU == 0:
        batch = [tensor for tensor in batch]
    else:
        batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
     ped_features) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    for _ in range(BEST_K):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                                  args.train_or_test, SPEED_TO_ADD, ped_features)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses


def check_accuracy(
        args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            if USE_GPU == 0:
                batch = [tensor for tensor in batch]
            else:
                batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, obs_ped_speed, pred_ped_speed, ped_features) = batch

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                                           args.train_or_test, SPEED_TO_ADD, ped_features)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel
            )
            ade = cal_ade(pred_traj_gt, pred_traj_fake)
            fde = cal_fde(pred_traj_gt, pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, ped_speed, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            mask_sum += (pred_traj_gt.size(1) * PRED_LEN)
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= NUM_SAMPLE_CHECK:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj

    generator.train()
    return metrics


def cal_l2_losses(
        pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel
):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    return ade


def cal_fde(pred_traj_gt, pred_traj_fake):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return fde


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
