import torch
import numpy as np
from constants import *
from utils import get_dataset_name


def get_traj(trajectories, sequences):
    print("Enter the sequence you want to visualize from:", sequences)
    seq_start = int(input("Enter the sequence start: "))
    seq_end = int(input("Enter the sequence end:"))
    positions = trajectories[:, seq_start:seq_end, :]
    return positions


def get_distance(trajectories):
    euclid_distance = []
    for a, b in zip(trajectories[:, :], trajectories[1:, :]):
        dist = torch.pairwise_distance(a, b)
        dist = dist.detach().numpy()
        euclid_distance.append(dist.reshape(1, -1))
    euclid_distance = torch.from_numpy(np.concatenate(euclid_distance, axis=0)).type(torch.float)
    return euclid_distance


def inverse_sigmoid(speed, max_speed):
    inv = torch.log((speed / (1 - speed)))
    print("The current speeds are: ", inv/max_speed)


def get_speed_from_distance(distance):
    traveling_speed = distance / FRAMES_PER_SECOND
    sigmoid_speed = torch.sigmoid(traveling_speed)
    return sigmoid_speed


def get_max_speed(path):
    if path == "eth":
        return ETH_MAX_SPEED
    elif path == "hotel":
        return HOTEL_MAX_SPEED
    elif path == "zara1":
        return ZARA1_MAX_SPEED
    elif path == "zara2":
        return ZARA2_MAX_SPEED
    elif path == "univ":
        return UNIV_MAX_SPEED


def verify_speed(traj, sequences):
    dataset_name = get_dataset_name(TEST_DATASET_PATH)
    max_speed = get_max_speed(dataset_name)
    traj = get_traj(traj, sequences)
    dist = get_distance(traj)
    speed = get_speed_from_distance(dist)
    # We calculate inverse sigmoid to verify the speed
    inverse_sigmoid(speed, max_speed)