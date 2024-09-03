import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

data = np.load('data/generated_trajectories.npy')


def analyze_trajectories(data):
    num_trajectories, num_points, _ = data.shape
    
    mean_lon = np.mean(data[:, :, 0])
    mean_lat = np.mean(data[:, :, 1])
    mean_speed = np.mean(data[:, :, 2])

    std_lon = np.std(data[:, :, 0])
    std_lat = np.std(data[:, :, 1])
    std_speed = np.std(data[:, :, 2])

    
    ci_lon = stats.t.interval(0.95, num_trajectories*num_points-1, loc=mean_lon, scale=std_lon/np.sqrt(num_trajectories*num_points))
    ci_lat = stats.t.interval(0.95, num_trajectories*num_points-1, loc=mean_lat, scale=std_lat/np.sqrt(num_trajectories*num_points))
    ci_speed = stats.t.interval(0.95, num_trajectories*num_points-1, loc=mean_speed, scale=std_speed/np.sqrt(num_trajectories*num_points))

    
    range_lon = np.max(data[:, :, 0]) - np.min(data[:, :, 0])
    range_lat = np.max(data[:, :, 1]) - np.min(data[:, :, 1])
    range_speed = np.max(data[:, :, 2]) - np.min(data[:, :, 2])
    range_heading = np.max(data[:, :, 3]) - np.min(data[:, :, 3])
    
    print(f"Longitude - Mean: {mean_lon:.4f}, CI: {ci_lon}, Range: {range_lon:.4f}")
    print(f"Latitude - Mean: {mean_lat:.4f}, CI: {ci_lat}, Range: {range_lat:.4f}")
    print(f"Speed - Mean: {mean_speed:.4f}, CI: {ci_speed}, Range: {range_speed:.4f}")
    print(f"Heading - Mean: {mean_heading:.4f}, CI: {ci_heading}, Range: {range_heading:.4f}")
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    sns.histplot(data[:, :, 0].flatten(), kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Longitude Distribution')
    sns.histplot(data[:, :, 1].flatten(), kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Latitude Distribution')
    sns.histplot(data[:, :, 2].flatten(), kde=True, ax=axs[1, 0])
    axs[1, 0].set_title('Speed Distribution')
    sns.histplot(data[:, :, 3].flatten(), kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('Heading Distribution')
    plt.tight_layout()
    plt.savefig('trajectory_distributions.png')
    plt.close()
    
    # 6. Plot trajectories on map
    plt.figure(figsize=(10, 10))
    for i in range(num_trajectories):
        plt.plot(data[i, :, 0], data[i, :, 1], alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectory Map')
    plt.savefig('trajectory_map.png')
    plt.close()
    
    # 7. Calculate and plot speed vs. heading
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, :, 3].flatten(), data[:, :, 2].flatten(), alpha=0.1)
    plt.xlabel('Heading')
    plt.ylabel('Speed')
    plt.title('Speed vs. Heading')
    plt.savefig('speed_vs_heading.png')
    plt.close()

# Run the analysis
analyze_trajectories(data)

# Additional analysis for control
def analyze_control_parameters(data):
    # Calculate acceleration and turn rate
    acceleration = np.diff(data[:, :, 2], axis=1)
    turn_rate = np.diff(data[:, :, 3], axis=1)
    
    # Print statistics
    print(f"Acceleration - Mean: {np.mean(acceleration):.4f}, Std: {np.std(acceleration):.4f}")
    print(f"Turn Rate - Mean: {np.mean(turn_rate):.4f}, Std: {np.std(turn_rate):.4f}")
    
    # Plot distributions
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(acceleration.flatten(), kde=True, ax=axs[0])
    axs[0].set_title('Acceleration Distribution')
    sns.histplot(turn_rate.flatten(), kde=True, ax=axs[1])
    axs[1].set_title('Turn Rate Distribution')
    plt.tight_layout()
    plt.savefig('control_distributions.png')
    plt.close()

analyze_control_parameters(data)