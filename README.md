# **CGANbasedTrajectoryPrediction**

* This project aims to generate realistic trajectories using Conditional GAN architecture with speed as an additional condition.

**To reproduce the project in virtual env, run the following command:**

Initially, clone the repo:
````
git clone https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction.git
````

````
conda create -n env_name python=3.7
````
The code was tested with python version 3.7.6.

This command will recognize the base package.
````
python CGANbasedTrajectoryPrediction/setup.py install
````

To install all dependencies, run:
````
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
````

To train the model, run:
````
python CGANbasedTrajectoryPrediction/scripts/train.py --train_path provide_train_path --val_path provide_val_path
````

To evaluate the model with no speed, run:
````
python CGANbasedTrajectoryPrediction/scripts/evaluate_model.py --test_path provide_test_path
````

To evaluate the model with additional speed, run:
````
python CGANbasedTrajectoryPrediction/scripts/evaluate_model.py --speed_to_add 0.5
````
**Note:** The speed value should be 0 < speed > 1
