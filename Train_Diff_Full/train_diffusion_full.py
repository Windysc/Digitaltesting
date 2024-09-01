import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
        )
        self.decoder = nn.Sequential(
            self.conv_block(256, 128),
            self.conv_block(128, 64),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

def augment_data(data, num_augmented=1000):
    augmented_data = []
    for _ in range(num_augmented):
        idx = np.random.randint(0, data.shape[0])
        traj = data[idx]
        noise = np.random.normal(0, 0.01, traj.shape)
        augmented_data.append(traj + noise)
    return np.array(augmented_data)


def train_model(model, data, epochs=3000, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = torch.FloatTensor(data[i:i+batch_size]).transpose(1, 2)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def generate_trajectories(model, num_trajectories=200):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_trajectories, 2, 91)
        generated = model(noise).transpose(1, 2).numpy()
    return generated

def visualize_trajectories(trajectories, num_to_plot=5, save_path=None):
    fig, axs = plt.subplots(1, num_to_plot, figsize=(20, 4))
    for i in range(num_to_plot):
        traj = trajectories[i]
        axs[i].plot(traj[:, 0], traj[:, 1])
        axs[i].set_title(f'Trajectory {i+1}')
        axs[i].axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_trajectories(trajectories, save_path):
    np.save(save_path, trajectories)

if __name__ == '__main__':
    output_dir = "trajectory_generation_output"
    os.makedirs(output_dir, exist_ok=True)

    data = np.load("/home/junze/.jupyter/Data transfer and loading/dataset_1.csv.npy")

    augmented_data = data

    model = SimpleUNet()
    train_model(model, augmented_data)

    model_save_path = os.path.join(output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    generated_trajectories = generate_trajectories(model)

    trajectories_save_path = os.path.join(output_dir, "generated_trajectoriesD.npy")
    save_trajectories(generated_trajectories, trajectories_save_path)
    print(f"Generated trajectories saved to {trajectories_save_path}")
    
    plot_save_path = os.path.join(output_dir, "generated_trajectoriesD_plot.png")
    visualize_trajectories(generated_trajectories, save_path=plot_save_path)
    print(f"Generated trajectories plot saved to {plot_save_path}")

    config = {
        "data_shape": data.shape,
        "augmented_data_shape": augmented_data.shape,
        "generated_trajectories_shape": generated_trajectories.shape,
        "model_architecture": str(model),
    }
    config_save_path = os.path.join(output_dir, "generation_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration and metadata saved to {config_save_path}")

    print("Trajectory generation process completed. Check the output directory for results.")