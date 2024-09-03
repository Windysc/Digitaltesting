import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

data = np.load('/home/junze/.jupyter/Train_VAE_full/datatrained_{n}x.npy')



def lat_lon_to_meters(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to meters using Equirectangular projection.
    """
    R = 6371000  # Earth's radius in meters
    x = R * np.radians(lon + ref_lon) * np.cos(np.radians(ref_lat))
    y = R * np.radians(lat + ref_lat)
    return x, y

def analyze_trajectories(data):
    num_trajectories, num_points, num_features = data.shape
    
    # Convert lat/lon to meters
    ref_lat, ref_lon = np.mean(data[:, 0, 1]), np.mean(data[:, 0, 0])
    data_meters = data.copy()
    for i in range(num_trajectories):
        data_meters[i, :, 0], data_meters[i, :, 1] = lat_lon_to_meters(data[i, :, 1], data[i, :, 0], ref_lat, ref_lon)
    
    # 1. Calculate average trajectory and bounds
    avg_trajectory = np.mean(data_meters, axis=0)
    std_trajectory = np.std(data_meters, axis=0)
    
    # Calculate upper and lower bounds (2 standard deviations)
    upper_bound = avg_trajectory + 2 * std_trajectory
    lower_bound = avg_trajectory - 2 * std_trajectory
    
    # 2. Plot trajectories with average and bounds
    plt.figure(figsize=(12, 8))
    for i in range(num_trajectories):
        plt.plot(data_meters[i, :, 0], data_meters[i, :, 1], alpha=0.1, color='gray')
    
    plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(upper_bound[:, 0], upper_bound[:, 1], color='red', linestyle='--', label='Upper Bound')
    plt.plot(lower_bound[:, 0], lower_bound[:, 1], color='green', linestyle='--', label='Lower Bound')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Trajectory Map with Average and Bounds')
    plt.legend()
    plt.savefig('trajectory_map_with_stats_meters.png')
    plt.close()
    
    # 3. Calculate and print statistics for each feature
    feature_names = ['X (meters)', 'Y (meters)', 'Speed', 'Heading']
    for i in range(num_features):
        feature_data = data_meters[:, :, i].flatten()
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        ci = stats.t.interval(0.95, len(feature_data)-1, loc=mean, scale=std/np.sqrt(len(feature_data)))
        range_val = np.max(feature_data) - np.min(feature_data)
        
        print(f"{feature_names[i]}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  95% CI: {ci}")
        print(f"  Range: {range_val:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print()
    
    # 4. Visualize distributions
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    for i in range(num_features):
        sns.histplot(data_meters[:, :, i].flatten(), kde=True, ax=axs[i//2, i%2])
        axs[i//2, i%2].set_title(f'{feature_names[i]} Distribution')
    plt.tight_layout()
    plt.savefig('feature_distributions_meters.png')
    plt.close()

    return avg_trajectory, upper_bound, lower_bound, data_meters

# Run the analysis
avg_trajectory, upper_bound, lower_bound, data_meters = analyze_trajectories(data)


def lat_lon_to_meters1(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to meters using Equirectangular projection.
    """
    R = 6371000  # Earth's radius in meters
    x = R * np.radians(lon + ref_lon) * np.cos(np.radians(ref_lat))
    y = R * np.radians(lat + ref_lat)
    return x, y

def analyze_trajectories1(data):
    num_trajectories, num_points, _ = data.shape
    
    # Calculate reference latitude and longitude
    ref_lat, ref_lon = np.mean(data[:, 0, 1]), np.mean(data[:, 0, 0])
    print(f"Reference Latitude: {ref_lat:.6f}")
    print(f"Reference Longitude: {ref_lon:.6f}")
    
    # Convert lat/lon to meters
    data_meters = np.zeros((num_trajectories, num_points, 2))
    for i in range(num_trajectories):
        data_meters[i, :, 0], data_meters[i, :, 1] = lat_lon_to_meters1(data[i, :, 1], data[i, :, 0], ref_lat, ref_lon)
    
    # Flatten the data for statistical analysis
    x_meters = data_meters[:, :, 0].flatten()
    y_meters = data_meters[:, :, 1].flatten()
    
    # Calculate statistics using scipy.stats.describe
    x_stats = stats.describe(x_meters)
    y_stats = stats.describe(y_meters)
    
    # Print statistics
     # Print statistics
    print("\nX coordinate (meters) statistics:")
    print(f"  Mean: {x_stats.mean:.2f}")
    print(f"  Standard deviation: {np.sqrt(x_stats.variance):.2f}")
    print(f"  Minimum: {x_stats.minmax[0]:.2f}")
    print(f"  Maximum: {x_stats.minmax[1]:.2f}")
    print(f"  Range: {x_stats.minmax[1] - x_stats.minmax[0]:.2f}")
    
    print("\nY coordinate (meters) statistics:")
    print(f"  Mean: {y_stats.mean:.2f}")
    print(f"  Standard deviation: {np.sqrt(y_stats.variance):.2f}")
    print(f"  Minimum: {y_stats.minmax[0]:.2f}")
    print(f"  Maximum: {y_stats.minmax[1]:.2f}")
    print(f"  Range: {y_stats.minmax[1] - y_stats.minmax[0]:.2f}")
    
    # Calculate average trajectory and bounds
    avg_trajectory = np.mean(data_meters, axis=0)
    std_trajectory = np.std(data_meters, axis=0)
    
    # Calculate upper and lower bounds (2 standard deviations)
    upper_bound = avg_trajectory + 2 * std_trajectory
    lower_bound = avg_trajectory - 2 * std_trajectory
    
    avg_distance_upper = np.mean(np.sqrt(np.sum((upper_bound - avg_trajectory)**2, axis=1)))
    avg_distance_lower = np.mean(np.sqrt(np.sum((lower_bound - avg_trajectory)**2, axis=1)))
    
    # Plot trajectories with average and bounds
    plt.figure(figsize=(12, 8))
    for i in range(num_trajectories):
        plt.plot(data_meters[i, :, 0], data_meters[i, :, 1], alpha=0.1, color='gray')
    
    plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(upper_bound[:, 0], upper_bound[:, 1], color='red', linestyle='--', label='Upper Bound')
    plt.plot(lower_bound[:, 0], lower_bound[:, 1], color='green', linestyle='--', label='Lower Bound')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Trajectory Map with Average and Bounds')
    plt.legend()
    plt.savefig('trajectory_map_with_stats_meters.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(upper_bound[:, 0], upper_bound[:, 1], color='red', linestyle='--', label='Upper Bound')
    plt.plot(lower_bound[:, 0], lower_bound[:, 1], color='green', linestyle='--', label='Lower Bound')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Zoomed-in View of Average Trajectory and Bounds')
    plt.legend()
    
    # Add annotations for distances
    mid_point = len(avg_trajectory) // 2
    plt.annotate(f'Avg. distance: {avg_distance_upper:.2f}m', 
                 xy=(avg_trajectory[mid_point, 0], avg_trajectory[mid_point, 1]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    
    plt.annotate(f'Avg. distance: {avg_distance_lower:.2f}m', 
                 xy=(avg_trajectory[mid_point, 0], avg_trajectory[mid_point, 1]),
                 xytext=(10, -10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    
    # Adjust the plot limits to zoom in
    x_range = np.max(avg_trajectory[:, 0]) - np.min(avg_trajectory[:, 0])
    y_range = np.max(avg_trajectory[:, 1]) - np.min(avg_trajectory[:, 1])
    plt.xlim(np.min(avg_trajectory[:, 0]) - 0.1*x_range, np.max(avg_trajectory[:, 0]) + 0.1*x_range)
    plt.ylim(np.min(avg_trajectory[:, 1]) - 0.1*y_range, np.max(avg_trajectory[:, 1]) + 0.1*y_range)
    
    plt.savefig('zoomed_trajectory_with_bounds.png')
    plt.close()

# Run the analysis
analyze_trajectories1(data)