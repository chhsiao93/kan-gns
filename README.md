# kan-gns
`source build_venv_frontera.sh` to create a virtual environment for TACC Frontera. The script will also download the waterdrop sample for training.

`source start_venv.sh` to load the required module on Frontera and activate the virtual environment.

`python gnn_train.py` to train the KAN-GNS model. To resume training a model, chaning the `total_steps` to the checkpoint number of the model. For example, `checkpoint_1000.pt` is the model you want to continue to train. Change the line in `gnn_train.py` to `total_steps=1000` and run `python gnn_train.py`, it will start training the model from 1000 steps.

`python gnn_rollout.py` to plot the train history and animation. You can specify which model to rollout by changing `checkpoint_name = 'checkpoint_xxx'` 
