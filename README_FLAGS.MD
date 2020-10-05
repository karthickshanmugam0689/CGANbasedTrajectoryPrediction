#Flag descriptions:

- H_DIM: Hidden dimensions used for both encoder and decoder LSTM
- EMBEDDING_DIM: Embedding dimension
- MLP_DIM: MLP dimension
- NOISE_DIM: Gaussian noise distribution dimension
- G_LEARNING_RATE, D_LEARNING_RATE: Learning rate of generator and discriminator model
- NUM_LAYERS = Number of layers we use in LSTM Encoder and DECODER network
- DROPOUT = Dropout to be used in the model
- NUM_EPOCHS = Number of epochs the model is to be trained. 

# Pooling Module
- POOLING_TYPE = Flag used to pass the output of encoder to social speed pooling module 
- DECODER_TIMESTEP_POOLING = For every timestep decoder predicts, it is possible to pool the predicted value with the help of this flag.

The following flags are used to anaylze the maximum speed w.r.t the dataset. Each dataset has different max speed value. 
ETH = 1 --> Multiplying the user provided speed with the maximum speed of ETH dataset
Similarly other flags work (ZARA01, ZARA02, HOTEL, UNI) 

- OBS_LEN, PRED_LEN = Observed and predicted length of the trajectories
- CHECKPOINT_NAME = Path of the checkpoint at which the weights needs to be stored.
- BATCH = Batch size
- BATCH_NORM = Batch normalization option
- ACTIVATION = We use leakyrelu activation function.
- G_STEPS, D_STEPS = The number of feed-forward and backward passes for generator and discriminator respectively.
- NUM_SAMPLES = Number of samples to test the model
- NOISE = Noise to be added.


# SPEED CONTROL FLAGS
- TRAIN_METRIC = We utilize the next speed information from the ground truth to train the model. To differentiate the ground_truth and simulated speeds, we utilize "train_metric" and "test_metric" flags. This value always holds 1. 
- TEST_METRIC = To check the model with ground truth speed, use 0. To simulate the model with user-defined speed, use 1.  
- The following flags are applicable only if the test_metric = 1
- ADD_SPEED_EVERY_FRAME = This flag adds constant speed to every predicted frame. If this flag is 1, then mention the speed to be added in SPEED_TO_ADD flag.
SPEED_TO_ADD = Speed to be added at each frame. The value should be between 0 and 1
- STOP_PED = This flag stops all the pedestrians. This implies that a speed value of 0 is imposed on the pedestrians.
- CONSTANT_SPEED_FOR_ALL_PED = This flags helps to 
- ADD_SPEED_PARTICULAR_FRAME = False
   FRAMES_TO_ADD_SPEED = []  # Provide a value between 0 to length of (predicted traj-1)
- MAX_SPEED = Maximum possible speed