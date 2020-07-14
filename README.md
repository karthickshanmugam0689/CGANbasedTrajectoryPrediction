# **CGANbasedTrajectoryPrediction**

* This project aims to generate realistic trajectories using Conditional GAN architecture with speed as an additional condition.

**To reproduce the project, run the following command:**

Initially, clone the repo:
````
git clone https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction.git
````

This command will recognize the base package.
````
python CGANbasedTrajectoryPrediction/setup.py install
````

To install all dependencies, run:
````
conda install --yes --file requirements.txt
````

To train the model, run:
````
python CGANbasedTrajectoryPrediction/scripts/train.py
````

To evaluate the model with no speed, run:
````
python CGANbasedTrajectoryPrediction/scripts/evaluate_model.py
````

To evaluate the model with additional speed, run:
````
python CGANbasedTrajectoryPrediction/scripts/evaluate_model.py --speed_to_add 0.5
````
**Note:** The speed value should be 0 <= speed >= 1
