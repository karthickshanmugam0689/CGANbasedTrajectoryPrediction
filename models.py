import torch
import torch.nn as nn
from constants import *
import math
from losses import relative_to_abs


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.spatial_embedding = nn.Sequential(nn.Linear(3, self.embedding_dim*2), nn.LeakyReLU(), nn.Linear(self.embedding_dim*2, self.embedding_dim))
        self.num_layers = NUM_LAYERS
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers, dropout=DROPOUT)

    def init_hidden(self, batch):
        if USE_GPU:
            c_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            h_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            c_s = torch.zeros(self.num_layers, batch, self.h_dim)
            h_s = torch.zeros(self.num_layers, batch, self.h_dim)
        return c_s, h_s

    def forward(self, obs_traj, obs_ped_speed):
        batch = obs_traj.size(1)
        embedding_input = torch.cat([obs_traj, obs_ped_speed], dim=2)
        traj_speed_embedding = self.spatial_embedding(embedding_input.contiguous().view(-1, 3))
        obs_traj_embedding = traj_speed_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h.squeeze(dim=0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inverseSigmoid(x):
    return math.log(x / (1 - x))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.dropout = DROPOUT
        self.num_layers = NUM_LAYERS
        self.mlp_dim = MLP_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers, dropout=self.dropout)

        self.social_speed_pooling = SocialSpeedPoolingModule()

        mlp_dims = [H_DIM + BOTTLENECK_DIM, MLP_DIM, H_DIM]
        self.mlp = make_mlp(mlp_dims, activation='relu', dropout=self.dropout)
        self.spatial_embedding = nn.Sequential(nn.Linear(3, self.embedding_dim*2), nn.LeakyReLU(), nn.Linear(self.embedding_dim*2, self.embedding_dim))
        self.hidden2pos = nn.Linear(self.h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, speed_to_add, pred_ped_speed, train_or_test):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        if train_or_test == 0:
            last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :]], dim=1)
        else:
            next_speed = speed_control(pred_ped_speed[0, :, :], SPEED_TO_ADD, seq_start_end)
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
                    speed = speed_control(pred_ped_speed[id+1, :, :], SPEED_TO_ADD, seq_start_end, id=id)
            embedding_input = rel_pos
            decoder_input = torch.cat([embedding_input, speed], dim=1)
            decoder_input = self.spatial_embedding(decoder_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            if DECODER_TIMESTEP_POOLING:
                decoder_h_state = state_tuple[0].squeeze(dim=0)
                sspm = self.social_speed_pooling(decoder_h_state, seq_start_end, train_or_test, speed_to_add,
                                                 "decoder", curr_pos, speed)
                decoder_h_state = self.mlp(torch.cat([decoder_h_state, sspm], dim=1))
                state_tuple = (decoder_h_state.unsqueeze(dim=0), state_tuple[1])
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class SocialSpeedPoolingModule(nn.Module):
    """The pooling module takes the speed of the pedestrians each other approaching into account"""
    def __init__(self):
        super(SocialSpeedPoolingModule, self).__init__()
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        mlp_input = [self.h_dim + self.embedding_dim, 512, self.bottleneck_dim]

        self.mlp_pre_pool = make_mlp(mlp_input, activation='leakyrelu', dropout=DROPOUT)
        self.pos_embedding = nn.Sequential(nn.Linear(3, self.embedding_dim*2), nn.LeakyReLU(), nn.Linear(self.embedding_dim*2, self.embedding_dim))

    def forward(self, h_states, seq_start_end, train_or_test, speed_to_add, encOrDec, last_pos, speed, ped_features=None):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states[start:end]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)

            if encOrDec == "encoder":
                if train_or_test == 0:
                    social_features_with_speed = ped_features[start:end, 0:num_ped, :].contiguous().view(-1, 3)
                else:
                    social_features = ped_features[start:end, 0:num_ped, :2].contiguous().view(-1, 2)
                    social_features_with_speed = torch.cat([social_features,
                                                            speed[start:end].view(-1, 1).repeat(num_ped, 1)], dim=1)
                    a = social_features_with_speed.view(num_ped, num_ped, 3)
            else:
                feature = torch.cat([last_pos[start:end], speed[start:end]], dim=1)
                if train_or_test == 1:
                    speed = speed_control(speed, SPEED_TO_ADD, seq_start_end)
                    feature = torch.cat([last_pos[start:end], speed[start:end]], dim=1)
                curr_end_pos_1 = feature.repeat(num_ped, 1)
                curr_end_pos_2 = feature.unsqueeze(dim=1).repeat(1, num_ped, 1).view(-1, 3)
                social_features = curr_end_pos_1[:, :2] - curr_end_pos_2[:, :2]
                social_features_with_speed = torch.cat([social_features, curr_end_pos_1[:, 2].view(-1, 1)], dim=1)

            # POSITION SPEED Pooling
            position_feature_embedding = self.pos_embedding(social_features_with_speed.contiguous().view(-1, 3))
            pos_mlp_input = torch.cat([repeat_hstate.view(-1, self.h_dim), position_feature_embedding.view(-1, self.embedding_dim)], dim=1)
            pos_attn_h = self.mlp_pre_pool(pos_mlp_input)
            curr_pool_h = pos_attn_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


def speed_control(pred_traj_first_speed, speed_to_add, seq_start_end, id=None):
    """This method represents the CONTROL MODULE in the paper. Using this module, user can add
    speed at one/more frames, stop the pedestrians and so on"""
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if ETH:
            speed_to_add = ETH_MAX_SPEED * SPEED_TO_ADD
        if HOTEL:
            speed_to_add = HOTEL_MAX_SPEED * SPEED_TO_ADD
        if UNIV:
            speed_to_add = UNIV_MAX_SPEED * SPEED_TO_ADD
        if ZARA1:
            speed_to_add = ZARA1_MAX_SPEED * SPEED_TO_ADD
        if ZARA2:
            speed_to_add = ZARA2_MAX_SPEED * SPEED_TO_ADD

        if ADD_SPEED_EVERY_FRAME:
            # To add an additional speed for each pedestrain and every frame
            for a in range(start, end):
                current_speed = inverseSigmoid(pred_traj_first_speed[a])
                if sigmoid(current_speed + speed_to_add) < 1:
                    pred_traj_first_speed[a] = sigmoid(current_speed + speed_to_add)
                else:
                    pred_traj_first_speed[a] = MAX_SPEED
        elif STOP_PED:
            # To stop all pedestrians
            speed_to_add = 0
            for a in range(start, end):
                pred_traj_first_speed[a] = sigmoid(speed_to_add)
        elif CONSTANT_SPEED_FOR_ALL_PED:
            # To make all pedestrians travel at same and constant speed throughout
            for a in range(start, end):
                pred_traj_first_speed[a] = sigmoid(speed_to_add)
        elif ADD_SPEED_PARTICULAR_FRAME and len(FRAMES_TO_ADD_SPEED) > 0:
            # Add speed to particular frame for all pedestrian
            sorted_frames = FRAMES_TO_ADD_SPEED.sort()
            for frames in sorted_frames:
                if id == frames and id != 0:
                    for a in range(start, end):
                        pred_traj_first_speed[a] = pred_traj_first_speed[a] + sigmoid(speed_to_add)
                else:
                    pred_traj_first_speed[a] = pred_traj_first_speed[a]

    return pred_traj_first_speed.view(-1, 1)


class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.social_speed_pooling = SocialSpeedPoolingModule()
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        self.mlp_dim = MLP_DIM
        self.noise_dim = NOISE_DIM
        self.dropout = DROPOUT

        if NOISE:
            output_dim = self.h_dim - self.noise_dim[0]
        else:
            output_dim = self.h_dim

        mlp_decoder_context_dims = [self.h_dim + self.bottleneck_dim, self.mlp_dim, output_dim]

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims, activation='leakyrelu', dropout=self.dropout)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj, train_or_test,
                speed_to_add, ped_features, user_noise=None):
        batch = obs_traj_rel.size(1)
        final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed)
        if POOLING_TYPE:
            if train_or_test == 1:
                simulated_ped_speed = speed_control(pred_ped_speed[0], SPEED_TO_ADD, seq_start_end)
                next_speed = simulated_ped_speed
            else:
                next_speed = pred_ped_speed[0]
            sspm = self.social_speed_pooling(final_encoder_h, seq_start_end, train_or_test, speed_to_add,
                                            "encoder", obs_traj[-1], next_speed, ped_features=ped_features)
            mlp_decoder_context_input = torch.cat([final_encoder_h, sspm], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h
        mlp_input = self.mlp_decoder_context(mlp_decoder_context_input)
        batch_noise_dim = (batch, ) + NOISE_DIM
        if USE_GPU:
            noise = torch.randn(*batch_noise_dim).cuda()
        else:
            noise = torch.randn(*batch_noise_dim)
        decoder_h = torch.cat([mlp_input, noise], dim=1).unsqueeze(dim=0)

        if USE_GPU:
            decoder_c = torch.zeros(NUM_LAYERS, batch, self.h_dim).cuda()
        else:
            decoder_c = torch.zeros(NUM_LAYERS, batch, self.h_dim)

        state_tuple = (decoder_h, decoder_c)

        decoder_out = self.decoder(
            obs_traj[-1],
            obs_traj_rel[-1],
            state_tuple,
            seq_start_end,
            speed_to_add,
            pred_ped_speed,
            train_or_test
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        # LOGGING THE OUTPUT OF ALL SEQUENCES TO TEST THE SPEED AND TRAJECTORIES
        if train_or_test == 1:
            outputs = {}
            for _, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                obs_test_traj = obs_traj[:, start:end, :]
                pred_test_traj_rel = pred_traj_fake_rel[:, start:end, :]
                pred_test_traj = relative_to_abs(pred_test_traj_rel, obs_test_traj[-1])
                speed_added = pred_ped_speed[0, start:end, :]
                seq_tuple = (start, end)
                outputs[seq_tuple] = pred_test_traj
        else:
            outputs = None
        return pred_traj_fake_rel, outputs


class TrajectoryDiscriminator(nn.Module):
    def __init__(self):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder()
        self.h_dim = H_DIM
        self.mlp_dim = MLP_DIM
        self.dropout = DROPOUT

        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation='relu',
            dropout=self.dropout
        )

    def forward(self, traj, traj_rel, ped_speed, seq_start_end=None):
        final_h = self.encoder(traj_rel, ped_speed)  # final layer of the encoder is returned
        scores = self.real_classifier(final_h.squeeze())  # mlp - 64 --> 1024 --> 1
        return scores
