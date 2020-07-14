# **CGANbasedTrajectoryPrediction**

* This project aims to generate realistic trajectories using Conditional GAN architecture with speed as an additional condition.

**To reproduce the project, run the following command:**

If using virtual environment, run:
````
python CGANbasedTrajectoryPrediction/setup.py develop
````

If running in global environment, run:
````
python CGANbasedTrajectoryPrediction/setup.py install
````

This command will recognize the base package. To install all dependencies, run:
````
pip install -r CGANbasedTrajectoryPrediction/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
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