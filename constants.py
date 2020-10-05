# DATASET OPTIONS
OBS_LEN = 8
PRED_LEN = 12
TRAIN_DATASET_PATH = 'datasets/eth/train'
VAL_DATASET_PATH = 'datasets/eth/val'
TEST_DATASET_PATH = 'datasets/eth/test'
CHECKPOINT_NAME = 'Checkpoints/ETH/model_checkpoint.pt'

# DATASET FLAGS FOR ANALYZING THE MAX SPEEDS.
ETH = 1
UNIV = 0
ZARA1 = 0
ZARA2 = 0
HOTEL = 0
ETH_MAX_SPEED = 3.902
HOTEL_MAX_SPEED = 2.3430
UNIV_MAX_SPEED = 2.1665
ZARA1_MAX_SPEED = 2.4873
ZARA2_MAX_SPEED = 2.2537

# PYTORCH DATA LOADER OPTIONS
NUM_WORKERS = 4
BATCH = 32
BATCH_NORM = False
ACTIVATION = 'relu'

# ENCODER DECODER HIDDEN DIMENSION OPTIONS
H_DIM = 32
H_DIM_DIS = 64

# HYPER PARAMETERS OPTIONS
G_LEARNING_RATE, D_LEARNING_RATE = 1e-3, 1e-3
NUM_LAYERS = 1
DROPOUT = 0
NUM_EPOCHS = 200
CHECKPOINT_EVERY = 50
MLP_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = (8, )
DECODER_TIMESTEP_POOLING = False
L2_LOSS_WEIGHT = 1

NUM_ITERATIONS = 20000
POOLING_TYPE = True
USE_GPU = 0

# SPEED CONTROL FLAGS
TEST_METRIC = 0  # 0 for ground_truth speed. To simulate trajectories, change the flag to 1. This flag is used during testing and inference phase.
TRAIN_METRIC = 0  # Used for training the model with the ground truth

# ADD_SPEED_EVERY_FRAME, STOP_PED, CONSTANT_SPEED_FOR_ALL_PED, ADD_SPEED_PARTICULAR_FRAME - Only one flag out of the 4 can be activated at once.
ADD_SPEED_EVERY_FRAME = True  # SPEED_TO_ADD will be active if ADD_SPEED_EVERY_FRAME is True
SPEED_TO_ADD = 0.1

STOP_PED = False  # Makes the speed value as 0

CONSTANT_SPEED_FOR_ALL_PED = False  # CONSTANT_SPEED flag will be active only if CONSTANT_SPEED_FOR_ALL_PED is True
CONSTANT_SPEED = 0.5

ADD_SPEED_PARTICULAR_FRAME = False  # FRAMES_TO_ADD_SPEED flag will be active only if ADD_SPEED_PARTICULAR_FRAME is True
FRAMES_TO_ADD_SPEED = []  # Provide a value between 0 to length of (predicted traj-1)
MAX_SPEED = 0.9999

G_STEPS = 1
D_STEPS = 2
BEST_K = 10
PRINT_EVERY = 100
NUM_SAMPLES = 20
NUM_SAMPLES_CHECK = 5000
NOISE = True
NUM_SAMPLE_CHECK = 5000