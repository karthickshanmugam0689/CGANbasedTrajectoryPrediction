# CGANbasedTrajectoryPrediction

This project aims to generate realistic trajectories using Conditional GAN architecture with speed as an additional condition.

To reproduce the project, run the following command:

If using virtual environment, run:
python setup.py develop

If running in global environment, run:
python setup.py install

This command will recognize the base package. To install all dependencies, run:
pip install -r requirements.txt

If using windows OS, to install torch and torchvision, use the below command:
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html


To train the model, run:
python scripts/train.py
