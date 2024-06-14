import os
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
from matplotlib import animation
from effKAN_gnn import LearnedSimulator, OneStepDataset, RolloutDataset, oneStepMSE, rollout


prefix = 'waterdrop' # prefix for gif file
name = 'effKAN6' # gif filename
checkpoint_name = 'checkpoint_800' # which checkpoint to rollout

DATASET_NAME = "WaterDropSample"
OUTPUT_DIR = "./WaterDropSample"
data_path = OUTPUT_DIR

model_path = os.path.join("temp", "models", DATASET_NAME)
rollout_path = os.path.join("temp", "rollouts", DATASET_NAME)


params = {
    "noise": 3e-4,
    "batch_size": 4,
}

# load dataset
valid_dataset = OneStepDataset(data_path, "valid", noise_std=params["noise"])
valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False, pin_memory=True, num_workers=2)

simulator = LearnedSimulator()
simulator = simulator.cuda()
# load checkpoint
checkpoint = torch.load(f"temp/models/WaterDropSample/{checkpoint_name}.pt")
simulator.load_state_dict(checkpoint["model"])
train_loss_tuple = checkpoint['train_loss']
eval_loss_tuple = checkpoint['eval_loss']
train_loss_arr = np.array(train_loss_tuple)
eval_loss_arr = np.array(eval_loss_tuple)
fig, ax = plt.subplots(1,1)
ax.plot(train_loss_arr[:,0],train_loss_arr[:,1], label='train')
ax.plot(eval_loss_arr[:,0],eval_loss_arr[:,1], label='eval')
ax.set_xlabel('steps', fontsize=15)
ax.set_ylabel('rollout_loss', fontsize=15)
plt.yscale("log")
plt.ylim(1e-3,1e2)
plt.legend()
plt.savefig(f'{rollout_path}/loss_history.png',dpi=150)

eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, valid_dataset.metadata, params["noise"])
print(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")

rollout_dataset = RolloutDataset(data_path, "valid")
simulator.eval()
rollout_data = rollout_dataset[0]
rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params["noise"])
rollout_out = rollout_out.permute(1, 0, 2)



TYPE_TO_COLOR = {
    3: "black",
    0: "green",
    7: "magenta",
    6: "gold",
    5: "blue",
}


def visualize_prepare(ax, particle_type, position, metadata):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_info = [
        visualize_prepare(axes[0], particle_type, position_gt, metadata),
        visualize_prepare(axes[1], particle_type, position_pred, metadata),
    ]
    axes[0].set_title("Ground truth")
    axes[1].set_title("Prediction")

    plt.close()

    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for type_, line in points.items():
                mask = particle_type == type_
                line.set_data(position[step_i, mask, 0], position[step_i, mask, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(fig, update, frames=np.arange(0, position_gt.size(0)), interval=10, blit=True)

anim = visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data["position"], rollout_dataset.metadata)
anim.save(f"{rollout_path}/{prefix}_{name}.gif", dpi=200, writer=animation.PillowWriter(fps=25))

