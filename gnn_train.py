import os
import torch
import torch_geometric as pyg
import numpy as np
from effKAN_gnn import LearnedSimulator, OneStepDataset, RolloutDataset, optimizer_to, train


### Training

print(f"PyTorch has version {torch.__version__} with cuda {torch.version.cuda}")

print("CUDA is ",torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())


DATASET_NAME = "WaterDropSample"
OUTPUT_DIR = "./WaterDropSample"

data_path = OUTPUT_DIR
model_path = os.path.join("temp", "models", DATASET_NAME)
rollout_path = os.path.join("temp", "rollouts", DATASET_NAME)

params = {
    "epoch": 10,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "noise": 3e-4,
    "save_interval": 200,
    "eval_interval": 200,
    "rollout_interval": 200000,
}

# load dataset
train_dataset = OneStepDataset(data_path, "train", noise_std=params["noise"])
valid_dataset = OneStepDataset(data_path, "valid", noise_std=params["noise"])
train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, pin_memory=True, num_workers=2)
valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False, pin_memory=True, num_workers=2)
valid_rollout_dataset = RolloutDataset(data_path, "valid")


# build model
simulator = LearnedSimulator()
optimizer = torch.optim.AdamW(simulator.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))
total_steps=0
# total_steps = 186000
# load checkpoint to resume training
if total_steps != 0:
    checkpoint = torch.load(f"temp/models/WaterDropSample/checkpoint_{total_steps}.pt")
    simulator.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
simulator = simulator.cuda()
optimizer_to(optim=optimizer, device=torch.device("cuda"))
scheduler = scheduler
total_params = sum(p.numel() for p in simulator.parameters() if p.requires_grad)

print("Total parameters in the network:", total_params)
# train the model
train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list = train(model_path, params, simulator, train_loader, valid_loader,valid_dataset , valid_rollout_dataset, optimizer=optimizer, scheduler=scheduler, total_step=total_steps)
