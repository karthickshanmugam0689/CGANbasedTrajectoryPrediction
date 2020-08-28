import torch
import torch.nn as nn
from csgan.constants import *
import math
from csgan.utils import relative_to_abs, get_dset_path


def make_mlp(dim_list, activation='leakyrelu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape)
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.LSTM(EMBEDDING_DIM, H_DIM, NUM_LAYERS, dropout=DROPOUT)

        self.spatial_embedding = nn.Linear(3, EMBEDDING_DIM)

    def init_hidden(self, batch):
        if USE_GPU == 0:
            c_s = torch.zeros(NUM_LAYERS, batch, H_DIM)
            h_s = torch.zeros(NUM_LAYERS, batch, H_DIM)
        else:
            c_s = torch.zeros(NUM_LAYERS, batch, H_DIM).cuda()
            h_s = torch.zeros(NUM_LAYERS, batch, H_DIM).cuda()
        return (c_s, h_s)

    def forward(self, obs_traj, obs_ped_speed):
        batch = obs_traj.size(1)
        embedding_input = torch.cat([obs_traj.contiguous().view(-1, 2), obs_ped_speed.contiguous().view(-1, 1)], dim=1)
        traj_speed_embedding = self.spatial_embedding(embedding_input.contiguous().view(-1, 3))
        obs_traj_embedding = traj_speed_embedding.view(-1, batch, EMBEDDING_DIM)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.LSTM(EMBEDDING_DIM, DECODER_H_DIM, NUM_LAYERS, dropout=DROPOUT)

        self.pool_net = PoolHiddenNet()

        mlp_dims = [DECODER_H_DIM + BOTTLENECK_DIM, MLP_DIM, DECODER_H_DIM]
        self.mlp = make_mlp(
            mlp_dims,
            activation='leakyrelu',
            batch_norm=BATCH_NORM,
            dropout=DROPOUT
        )

        self.spatial_embedding = nn.Linear(3, EMBEDDING_DIM)
        self.hidden2pos = nn.Linear(DECODER_H_DIM, 2)
        self.embedding_dim = EMBEDDING_DIM
        self.h_dim = DECODER_H_DIM

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, pred_ped_speed, train_or_test):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        if train_or_test == 0:
            last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :]], dim=1)
        else:
            next_speed = speed_control(pred_ped_speed[0, :, :], 0, seq_start_end)
            last_pos_speed = torch.cat([last_pos_rel, next_speed], dim=1)
        decoder_input = self.spatial_embedding(last_pos_speed)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for id in range(PRED_LEN):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            if id+1 != PRED_LEN:
                if train_or_test == 0:
                    speed = pred_ped_speed[id+1, :, :]
                else:
                    speed = speed_control(pred_ped_speed[id+1, :, :], 0, seq_start_end)
            embedding_input = rel_pos
            decoder_input = torch.cat([embedding_input, speed], dim=1)
            decoder_input = self.spatial_embedding(decoder_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    def __init__(self):
        super(PoolHiddenNet, self).__init__()

        mlp_pre_dim = EMBEDDING_DIM*2  # concatenating_speed_dimension
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]

        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation='relu',
            batch_norm=BATCH_NORM,
            dropout=DROPOUT)

        self.pos_embedding = nn.Linear(3, EMBEDDING_DIM)
        self.embedding_dim = EMBEDDING_DIM

    def forward(self, h_states, seq_start_end, train_or_test, speed_to_add, ped_features):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states.view(-1, H_DIM)[start:end]

            curr_end_pos = ped_features[start:end, 0:num_ped, :3]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)

            # POSITION ATTENTION
            position_feature_embedding = self.pos_embedding(curr_end_pos.contiguous().view(-1, 3))
            pos_mlp_input = torch.cat([repeat_hstate.view(-1, self.embedding_dim),
                                       position_feature_embedding.view(-1, self.embedding_dim)], dim=1)
            pos_attn_h = self.mlp_pre_pool(pos_mlp_input)
            curr_pool_h = pos_attn_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


def speed_control(pred_traj_first_speed, speed_to_add, seq_start_end):
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        # Evenly distributed speed for all pedestrian in each sequence
        speed = 0.1
        for a in range(start, end):
            if sigmoid(speed) < 1:
                pred_traj_first_speed[a] = sigmoid(speed)
                speed += 0.1
            else:
                pred_traj_first_speed[a] = 1
                speed = 1
    return pred_traj_first_speed


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, embedding_dim_pooling=64
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.mlp_dim = MLP_DIM
        self.encoder_h_dim = ENCODER_H_DIM
        self.decoder_h_dim = DECODER_H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.noise_dim = NOISE_DIM
        self.num_layers = NUM_LAYERS
        self.noise_type = NOISE_TYPE
        self.noise_mix_type = NOISE_MIX_TYPE
        self.bottleneck_dim = BOTTLENECK_DIM

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.pool_net = PoolHiddenNet()

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = NOISE_DIM[0]

        # Decoder Hidden
        input_dim = ENCODER_H_DIM + BOTTLENECK_DIM

        mlp_decoder_context_dims = [input_dim, MLP_DIM, DECODER_H_DIM - self.noise_first_dim]

        self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=BATCH_NORM,
                dropout=DROPOUT
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
                self.noise_dim or self.pooling_type or
                self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj, train_or_test,
                speed_to_add, ped_features, user_noise=None):
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed)
        # Pool States
        if POOLING_TYPE:
            # SPEED AT WHICH NEXT POSITION NEEDS TO BE PREDICTED
            attn_pool = self.pool_net(final_encoder_h, seq_start_end, train_or_test,
                                      speed_to_add, ped_features)
            # concatenating pooling module output with encoder output and speed embedding
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), attn_pool], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        if USE_GPU == 0:
            decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim)
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
            pred_ped_speed,
            train_or_test
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        # LOGGING THE OUTPUT OF ALL SEQUENCES TO TEST
        if train_or_test == 1:
            for _, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                obs_test_traj = obs_traj[:, start:end, :]
                pred_test_traj_rel = pred_traj_fake_rel[:, start:end, :]
                pred_test_traj = relative_to_abs(pred_test_traj_rel, obs_test_traj[-1])
                speed_added = pred_ped_speed[0, start:end, :]
                print("Start end", start, end)
                print("speed after adding:", speed_added)
                print("speed", speed_to_add, "pred_test_traj", pred_test_traj)
                print("§§$$%&//(())*/-")

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(self):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder()

        real_classifier_dims = [H_DIM, MLP_DIM, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation='relu',
            batch_norm=BATCH_NORM,
            dropout=DROPOUT
        )

    def forward(self, traj, traj_rel, ped_speed, seq_start_end=None):
        final_h = self.encoder(traj_rel, ped_speed)  # final layer of the encoder is returned
        scores = self.real_classifier(final_h.squeeze())  # mlp - 64 --> 1024 --> 1
        return scores
