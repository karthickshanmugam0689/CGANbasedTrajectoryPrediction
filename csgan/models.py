import torch
import torch.nn as nn


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
    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
            dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        #obs_traj_speed = torch.cat([obs_traj, obs_ped_speed], dim=2)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    def __init__(
            self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, pooling_type='pool_net',
            neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding_with_speed = nn.Linear(128, embedding_dim)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, last_speed_pos):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        last_pos_embedding = self.spatial_embedding(last_pos_rel)
        decoder_input = torch.cat([last_pos_embedding, last_speed_pos], dim=1)
        decoder_input = self.spatial_embedding_with_speed(decoder_input)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            # if self.pool_every_timestep:
            #     decoder_h = state_tuple[0]
            #     pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, obs_ped_speed)
            #     decoder_h = torch.cat(
            #         [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
            #     decoder_h = self.mlp(decoder_h)
            #     decoder_h = torch.unsqueeze(decoder_h, 0)
            #     state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    def __init__(
            self, embedding_dim_pooling=128, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
            activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = 128
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim_pooling = 128

        mlp_pre_dim = embedding_dim_pooling + embedding_dim_pooling + embedding_dim_pooling  # concatenating_speed_dimension
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim_pooling)
        self.speed_embedding = nn.Linear(64, embedding_dim_pooling)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos, end_pos_speed):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_ped_speed = end_pos_speed[start:end]
            curr_ped_speed_1 = curr_ped_speed.repeat(num_ped, 1)

            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            curr_ped_speed_embedding = self.speed_embedding(curr_ped_speed_1)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1, curr_ped_speed_embedding], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, skip=1, embedding_dim_pooling=128
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.skip = skip
        self.speed_embedding_sigmoid_layer = nn.Sigmoid()
        self.speed_embedding_layer = nn.Linear(8, 64)
        self.embedding_dim_pooling = embedding_dim_pooling

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim_pooling=self.embedding_dim_pooling,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
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

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, user_noise=None):
        batch = obs_traj_rel.size(1)
        obs_ped_speed = self.speed_embedding_sigmoid_layer(obs_ped_speed)
        obs_ped_speed_embedding = self.speed_embedding_layer(obs_ped_speed)
        obs_ped_speed = obs_ped_speed_embedding.unsqueeze(dim=2)
        obs_ped_speed = obs_ped_speed.permute(2, 1, 0)
        obs_ped_speed = obs_ped_speed.permute(0, 2, 1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        final_encoder_h_with_speed_embedding = torch.cat([final_encoder_h, obs_ped_speed], dim=2)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            end_speed_pos = obs_ped_speed[-1, :, :]
            pool_h = self.pool_net(final_encoder_h_with_speed_embedding, seq_start_end, end_pos, end_speed_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        # decoder_h = torch.cat([decoder_h, skip_condition_list], dim=1)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim)\
         #   .cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        last_speed_pos = obs_ped_speed[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
            last_speed_pos
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class EncoderDiscriminator(nn.Module):
    """This Encoder is part of TrajectoryDiscriminator"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0
    ):
        super(EncoderDiscriminator, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(3, embedding_dim)

    def init_hidden(self, batch):  # This is where we'll initialise our hidden state as
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim)
        )

    def forward(self, obs_traj, ped_speed):
        batch = obs_traj.size(1)
        ped_speed_embedding = ped_speed.unsqueeze(dim=2)
        ped_speed_embedding = ped_speed_embedding.permute(1, 0, 2)
        obs_traj_speed = torch.cat([obs_traj, ped_speed_embedding], dim=2)
        obs_traj_speed_embedding = self.spatial_embedding(obs_traj_speed.contiguous().view(-1, 3))
        obs_traj_speed_embedding = obs_traj_speed_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_speed_embedding, state_tuple)
        final_h = state[0]
        return final_h


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024, skip=1,
                 num_layers=1, activation='relu', batch_norm=True, dropout=0.0, d_type='local'):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.encoder_d = EncoderDiscriminator(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, ped_speed, seq_start_end=None):
        final_h = self.encoder_d(traj_rel, ped_speed)  # final layer of the encoder is returned
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)  # mlp - 64 --> 1024 --> 1
        return scores
