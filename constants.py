# DATASET OPTIONS
OBS_LEN = 8
PRED_LEN = 12
TRAIN_DATASET_PATH = 'C:/Users/visha/MasterThesis/CGANbasedTrajectoryPrediction/datasets/eth/train'  # FixMe: Replace the train_dataset path
VAL_DATASET_PATH = 'C:/Users/visha/MasterThesis/CGANbasedTrajectoryPrediction/datasets/eth/val'  # FixMe: Replace the val_dataset path
TEST_DATASET_PATH = 'C:/Users/visha/MasterThesis/CGANbasedTrajectoryPrediction/datasets/eth/test'  # FixMe: Replace the test_dataset path
CHECKPOINT_NAME = 'C:/Users/visha/MasterThesis/CGANbasedTrajectoryPrediction/Checkpoints/ETH_NEW/model_checkpoint.pt'  # FixMe: Replace the checkpoints path

# DATASET FLAGS FOR ANALYZING THE MAX SPEEDS.
# FixMe: Turn the corresponding dataset flags to 1 and others to 0
ETH = 1
UNIV = 0
ZARA1 = 0
ZARA2 = 0
HOTEL = 0

# MAX SPEEDS FOR EACH DATASETS
ETH_MAX_SPEED = 3.902
HOTEL_MAX_SPEED = 2.3430
UNIV_MAX_SPEED = 2.1665
ZARA1_MAX_SPEED = 2.4873
ZARA2_MAX_SPEED = 2.2537


# PYTORCH DATA LOADER OPTIONS
NUM_WORKERS = 4
BATCH = 32
BATCH_NORM = False
ACTIVATION = 'leakyrelu'

# MODEL DIMENSION OPTIONS
H_DIM = 32
H_DIM_DIS = 64
MLP_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = (8, )
L2_LOSS_WEIGHT = 1
MAX_NEAREST_PED = 3

# OTHER HYPER PARAMETERS OPTIONS
G_LEARNING_RATE, D_LEARNING_RATE = 1e-3, 1e-3
NUM_LAYERS = 1
DROPOUT = 0
NUM_EPOCHS = 200
CHECKPOINT_EVERY = 50
USE_GPU = 0
DECODER_TIMESTEP_POOLING = False
NUM_ITERATIONS = 20000
POOLING_TYPE = True

# SPEED CONTROL FLAGS
# FixMe: Depending on the type of testing, turn on the below flags. For adding speed every frame, turn the flag to "True" and others to false and so on.. More detailed explanation in Description.MD file
ADD_SPEED_EVERY_FRAME = True
SPEED_TO_ADD = 0.1
STOP_PED = False
CONSTANT_SPEED_FOR_ALL_PED = False
ADD_SPEED_PARTICULAR_FRAME = False
FRAMES_TO_ADD_SPEED = []  # Provide a value between 0 to length of (predicted traj-1)
MAX_SPEED = 0.9999

G_STEPS = 1
D_STEPS = 2
BEST_K = 10
PRINT_EVERY = 100
NUM_SAMPLES = 20
NUM_SAMPLES_CHECK = 5000
NOISE = True
TEST_METRIC = 0
TRAIN_METRIC = 0

NUM_SAMPLE_CHECK = 5000