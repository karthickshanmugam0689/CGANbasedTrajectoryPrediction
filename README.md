# **CGANbasedTrajectoryPrediction**  (WORK IS STILL IN PROGRESS - BETA VERSION)

* This project aims to generate realistic trajectories using Conditional GAN architecture with speed as an additional condition.
* The model proposed in this project can be used to simulate trajectories at different speeds

Pedestrian Simulation with Original Speed:
![OriginalSpeedPlot](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/master/Plots/Actual%20Speed.gif)

The output plot below indicates the distance traveled by the pedestrian with increase in Speed
![IncreasedSpeedPlot](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/master/Plots/Speed%200.4.gif)

**To reproduce the project, run the following command:**

Initially, clone the repo:
````
git clone https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction.git
````

````
python setup.py install
````

To install all dependencies, run:
````
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
````

Before training the model, give the absolute path of train, val and test dataset folder in constants.py file. Once changed,
run the following command:
````
python train.py
````

To evaluate the model with actual/real trajectory speed, run:
````
python evaluate_model.py
````

The control module is coded for
To evaluate the model with additional speed, run:
````
python CGANbasedTrajectoryPrediction/scripts/evaluate_model.py --speed_to_add 0.5
````
**Note:** The speed value should be 0 < speed > 1
