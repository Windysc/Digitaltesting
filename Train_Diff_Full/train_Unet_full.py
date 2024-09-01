import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def inspect_data(data):
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Standard deviation: {np.std(data)}")
    
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.plot(data[i, :, 0], data[i, :, 1])
        plt.title(f"Raw Trajectory {i+1}")
    plt.tight_layout()
    plt.savefig('Raw_Trajectories.png')
    plt.close()

# Updated normalization functions
def normalize_trajectories(trajectories):
    min_vals = np.min(trajectories, axis=(0, 1))
    max_vals = np.max(trajectories, axis=(0, 1))
    normalized = (trajectories - min_vals) / (max_vals - min_vals)
    return normalized, min_vals, max_vals

def denormalize_trajectories(normalized_trajectories, min_vals, max_vals):
    return normalized_trajectories * (max_vals - min_vals) + min_vals

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, time_dim=256, traj_length=91):
        super(SimpleUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, traj_length)
        )
        
        self.encoder = nn.Sequential(
            self.conv_block(in_channels + 1, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
        )
        self.decoder = nn.Sequential(
            self.conv_block(256, 128),
            self.conv_block(128, 64),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        )
        self.traj_length = traj_length

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, t):
        x = x.permute(0, 2, 1)
        t = t.float().unsqueeze(-1)
        t = self.time_mlp(t)
        t = t.unsqueeze(1)
        x = torch.cat([x, t], dim=1)
        features = self.encoder(x)
        output = self.decoder(features)
        return output.permute(0, 2, 1)  # Remove sigmoid activation

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def generate_trajectories(model, num_trajectories=200, timesteps=1000, traj_length=91):
    model.eval()
    device = next(model.parameters()).device
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    with torch.no_grad():
        x = torch.randn(num_trajectories, traj_length, 2).to(device)
        for i in tqdm(reversed(range(1, timesteps)), desc="Generating trajectories"):
            t = torch.full((num_trajectories,), i, dtype=torch.long, device=device)
            predicted_noise = model(x, t)
            alpha = alphas[i]
            alpha_cumprod = alphas_cumprod[i]
            beta = betas[i]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
    
    return x.cpu().numpy()

def visualize_trajectories(original_trajectories, generated_trajectories, num_to_plot=5):
    fig, axs = plt.subplots(2, num_to_plot, figsize=(20, 8))
    for i in range(num_to_plot):
        orig_traj = original_trajectories[i]
        axs[0, i].plot(orig_traj[:, 0], orig_traj[:, 1])
        axs[0, i].set_title(f'Original Trajectory {i+1}')
        axs[0, i].axis('equal')
        
        gen_traj = generated_trajectories[i]
        axs[1, i].plot(gen_traj[:, 0], gen_traj[:, 1])
        axs[1, i].set_title(f'Generated Trajectory {i+1}')
        axs[1, i].axis('equal')
    
    plt.tight_layout()
    plt.savefig('Trajectory_Comparison1.png')
    plt.close()

def save_generated_data(trajectories):
    np.save('generated_trajectoriesU2.npy', trajectories)

def train_model(model, data, epochs=1000, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    total_batches = len(data) // batch_size
    device = next(model.parameters()).device
    
    beta_schedule = linear_beta_schedule(1000).to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i in progress_bar:
            batch = torch.FloatTensor(data[i:i+batch_size]).to(device)
            optimizer.zero_grad()
            
            t = torch.randint(0, 1000, (batch.size(0),)).to(device)
            
            noise = torch.randn_like(batch)
            noisy_batch = batch + noise * torch.sqrt(beta_schedule[t].view(-1, 1, 1))
            
            predicted_noise = model(noisy_batch, t)
            
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / total_batches
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}')
    
    print("Training completed.")

if __name__ == "__main__":
    data = np.load('/home/junze/.jupyter/Data transfer and loading/dataset_2.csv.npy')
    
    inspect_data(data)
    
    normalized_data, min_vals, max_vals = normalize_trajectories(data)
    
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.plot(normalized_data[i, :, 0], normalized_data[i, :, 1])
        plt.title(f"Normalized Trajectory {i+1}")
    plt.tight_layout()
    plt.savefig('Normalized_Trajectories1.png')
    plt.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    traj_length = data.shape[1]
    model = SimpleUNet(in_channels=2, out_channels=2, time_dim=256, traj_length=traj_length).to(device)
    
    train_model(model, normalized_data)
    
    generated_trajectories = generate_trajectories(model, num_trajectories=data.shape[0], traj_length=traj_length)
    
    denormalized_trajectories = denormalize_trajectories(generated_trajectories, min_vals, max_vals)
    
    visualize_trajectories(data, denormalized_trajectories)
    save_generated_data(denormalized_trajectories)

    print(f"Generated trajectories shape: {denormalized_trajectories.shape}")
    print("Trajectories saved to 'generated_trajectories.npy'")
    print("Visualization saved to 'Trajectory_Comparison.png'")

    print(f"Original data mean: {np.mean(data)}")
    print(f"Original data std: {np.std(data)}")
    print(f"Generated data mean: {np.mean(denormalized_trajectories)}")
    print(f"Generated data std: {np.std(denormalized_trajectories)}")
    print(f"Min original value: {np.min(data)}")
    print(f"Max original value: {np.max(data)}")
    print(f"Min generated value: {np.min(denormalized_trajectories)}")
    print(f"Max generated value: {np.max(denormalized_trajectories)}")