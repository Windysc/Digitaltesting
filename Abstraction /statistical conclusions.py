import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.spatial.distance import mahalanobis



data = np.load('/home/junze/.jupyter/Digitaltesting/Ship_envre/downsampled_trajectory1.npy')



def lat_lon_to_meters(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to meters using Equirectangular projection.
    """
    R = 6371000  
    x = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y = R * np.radians(lat - ref_lat)
    return x, y

def smooth_trajectory(trajectory, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth the trajectory.
    """
    smoothed = np.zeros_like(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed[:, i] = savgol_filter(trajectory[:, i], window_length, polyorder)
    return smoothed

def analyze_trajectories2(data):
    num_trajectories, num_points, _ = data.shape
    
    # Calculate reference latitude and longitude
    ref_lat, ref_lon = np.mean(data[:, 0, 1]), np.mean(data[:, 0, 0])
    print(f"Reference Latitude: {ref_lat:.6f}")
    print(f"Reference Longitude: {ref_lon:.6f}")
    
    # Convert lat/lon to meters
    data_meters = np.zeros((num_trajectories, num_points, 2))
    for i in range(num_trajectories):
        data_meters[i, :, 0], data_meters[i, :, 1] = lat_lon_to_meters(data[i, :, 1], data[i, :, 0], ref_lat, ref_lon)
    
    # Save the data converted to meters
    np.save('downsampled_trajectory1_meters.npy', data_meters)
    
    # Calculate statistics for each time stamp
    avg_trajectory = np.mean(data_meters, axis=0)
    median_trajectory = np.median(data_meters, axis=0)
    lower_bound = np.percentile(data_meters, 2.5, axis=0)
    upper_bound = np.percentile(data_meters, 97.5, axis=0)
    
    # Smooth the trajectories
    avg_trajectory_smooth = smooth_trajectory(avg_trajectory)
    median_trajectory_smooth = smooth_trajectory(median_trajectory)
    lower_bound_smooth = smooth_trajectory(lower_bound)
    upper_bound_smooth = smooth_trajectory(upper_bound)
    
    # Print statistics for the first and last time stamps
    for i, label in enumerate(["First", "Last"]):
        idx = 0 if label == "First" else -1
        print(f"\n{label} time stamp statistics:")
        print(f"X coordinate (meters):")
        print(f"  Mean: {avg_trajectory_smooth[idx, 0]:.2f}")
        print(f"  Median: {median_trajectory_smooth[idx, 0]:.2f}")
        print(f"  95% Range: ({lower_bound_smooth[idx, 0]:.2f}, {upper_bound_smooth[idx, 0]:.2f})")
        print(f"Y coordinate (meters):")
        print(f"  Mean: {avg_trajectory_smooth[idx, 1]:.2f}")
        print(f"  Median: {median_trajectory_smooth[idx, 1]:.2f}")
        print(f"  95% Range: ({lower_bound_smooth[idx, 1]:.2f}, {upper_bound_smooth[idx, 1]:.2f})")
    
    # Plot trajectories with average and bounds
    plt.figure(figsize=(12, 8))
    for i in range(num_trajectories):
        plt.plot(data_meters[i, :, 0], data_meters[i, :, 1], alpha=0.1, color='gray')
    
    plt.plot(avg_trajectory_smooth[:, 0], avg_trajectory_smooth[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(median_trajectory_smooth[:, 0], median_trajectory_smooth[:, 1], color='cyan', linewidth=2, label='Median Trajectory')
    plt.plot(upper_bound_smooth[:, 0], upper_bound_smooth[:, 1], color='red', linestyle='--', label='97.5th Percentile')
    plt.plot(lower_bound_smooth[:, 0], lower_bound_smooth[:, 1], color='green', linestyle='--', label='2.5th Percentile')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Trajectory Map with Smoothed Average, Median, and 95% Range')
    plt.legend()
    plt.savefig('trajectory_map_with_smoothed_percentiles.png')
    plt.close()
    
    # Zoomed-in view
    plt.figure(figsize=(12, 8))
    plt.plot(avg_trajectory_smooth[:, 0], avg_trajectory_smooth[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(median_trajectory_smooth[:, 0], median_trajectory_smooth[:, 1], color='cyan', linewidth=2, label='Median Trajectory')
    plt.plot(upper_bound_smooth[:, 0], upper_bound_smooth[:, 1], color='red', linestyle='--', label='97.5th Percentile')
    plt.plot(lower_bound_smooth[:, 0], lower_bound_smooth[:, 1], color='green', linestyle='--', label='2.5th Percentile')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Zoomed-in View of Smoothed Trajectories with 95% Range')
    plt.legend()
    
    # Add annotations for average range width
    avg_range_width_x = np.mean(upper_bound_smooth[:, 0] - lower_bound_smooth[:, 0])
    avg_range_width_y = np.mean(upper_bound_smooth[:, 1] - lower_bound_smooth[:, 1])
    avg_range_width = np.sqrt(avg_range_width_x**2 + avg_range_width_y**2)
    
    mid_point = len(avg_trajectory_smooth) // 2
    plt.annotate(f'Avg. 95% range width: {avg_range_width:.2f}m', 
                 xy=(avg_trajectory_smooth[mid_point, 0], avg_trajectory_smooth[mid_point, 1]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    
    # Adjust the plot limits to zoom in
    x_range = np.max(avg_trajectory_smooth[:, 0]) - np.min(avg_trajectory_smooth[:, 0])
    y_range = np.max(avg_trajectory_smooth[:, 1]) - np.min(avg_trajectory_smooth[:, 1])
    plt.xlim(np.min(avg_trajectory_smooth[:, 0]) - 0.1*x_range, np.max(avg_trajectory_smooth[:, 0]) + 0.1*x_range)
    plt.ylim(np.min(avg_trajectory_smooth[:, 1]) - 0.1*y_range, np.max(avg_trajectory_smooth[:, 1]) + 0.1*y_range)
    
    plt.savefig('zoomed_trajectory_with_smoothed_percentiles.png')
    plt.close()

    return avg_trajectory_smooth, median_trajectory_smooth, lower_bound_smooth, upper_bound_smooth

def analyze_trajectories3(data):
    num_trajectories, num_points, _ = data.shape
    
    # Calculate reference latitude and longitude
    ref_lat, ref_lon = np.mean(data[:, 0, 1]), np.mean(data[:, 0, 0])
    print(f"Reference Latitude: {ref_lat:.6f}")
    print(f"Reference Longitude: {ref_lon:.6f}")
    
    # Convert lat/lon to meters
    data_meters = np.zeros((num_trajectories, num_points, 2))
    for i in range(num_trajectories):
        data_meters[i, :, 0], data_meters[i, :, 1] = lat_lon_to_meters(data[i, :, 1], data[i, :, 0], ref_lat, ref_lon)
    
    # Reshape data for easier statistical analysis
    data_reshaped = data_meters.reshape(num_trajectories, -1)
    
    # Calculate mean and covariance
    mean_trajectory = np.mean(data_reshaped, axis=0)
    cov_matrix = np.cov(data_reshaped, rowvar=False)
    
    # Calculate Mahalanobis distances
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distances = np.array([
        mahalanobis(trajectory, mean_trajectory, inv_cov_matrix)
        for trajectory in data_reshaped
    ])
    
    # Select boundary trajectories
    num_std = 3  # Adjust this to control the range of selected trajectories
    lower_bound_index = np.argmin(mahalanobis_distances)
    upper_bound_index = np.argmax(mahalanobis_distances)
    median_index = np.argmin(np.abs(mahalanobis_distances - np.median(mahalanobis_distances)))
    
    lower_trajectory = data_meters[lower_bound_index]
    upper_trajectory = data_meters[upper_bound_index]
    median_trajectory = data_meters[median_index]
    mean_trajectory = mean_trajectory.reshape(num_points, 2)
    
    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"Mean Mahalanobis distance: {np.mean(mahalanobis_distances):.2f}")
    print(f"Median Mahalanobis distance: {np.median(mahalanobis_distances):.2f}")
    print(f"Std dev of Mahalanobis distances: {np.std(mahalanobis_distances):.2f}")
    print(f"Min Mahalanobis distance: {np.min(mahalanobis_distances):.2f}")
    print(f"Max Mahalanobis distance: {np.max(mahalanobis_distances):.2f}")
    
    # Plot trajectories
    plt.figure(figsize=(12, 8))
    for i in range(num_trajectories):
        plt.plot(data_meters[i, :, 0], data_meters[i, :, 1], alpha=0.1, color='gray')
    
    plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color='blue', linewidth=2, label='Mean Trajectory')
    plt.plot(median_trajectory[:, 0], median_trajectory[:, 1], color='green', linewidth=2, label='Median Trajectory')
    plt.plot(upper_trajectory[:, 0], upper_trajectory[:, 1], color='red', linestyle='--', label='Upper Bound Trajectory')
    plt.plot(lower_trajectory[:, 0], lower_trajectory[:, 1], color='cyan', linestyle='--', label='Lower Bound Trajectory')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Trajectory Map with Statistical Boundaries')
    plt.legend()
    plt.savefig('trajectory_map_with_mahalanobis_boundaries.png')
    plt.close()
    
    # Zoomed-in view
    plt.figure(figsize=(12, 8))
    plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color='blue', linewidth=2, label='Mean Trajectory')
    plt.plot(median_trajectory[:, 0], median_trajectory[:, 1], color='green', linewidth=2, label='Median Trajectory')
    plt.plot(upper_trajectory[:, 0], upper_trajectory[:, 1], color='red', linestyle='--', label='Upper Bound Trajectory')
    plt.plot(lower_trajectory[:, 0], lower_trajectory[:, 1], color='cyan', linestyle='--', label='Lower Bound Trajectory')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Zoomed-in View of Trajectories with Statistical Boundaries')
    plt.legend()
    
    # Adjust the plot limits to zoom in
    x_range = np.max(mean_trajectory[:, 0]) - np.min(mean_trajectory[:, 0])
    y_range = np.max(mean_trajectory[:, 1]) - np.min(mean_trajectory[:, 1])
    plt.xlim(np.min(mean_trajectory[:, 0]) - 0.1*x_range, np.max(mean_trajectory[:, 0]) + 0.1*x_range)
    plt.ylim(np.min(mean_trajectory[:, 1]) - 0.1*y_range, np.max(mean_trajectory[:, 1]) + 0.1*y_range)
    
    plt.savefig('zoomed_trajectory_with_mahalanobis_boundaries.png')
    plt.close()

    return mean_trajectory, median_trajectory, lower_trajectory, upper_trajectory

# Run the analysis

def analyze_trajectories4(data):
    num_trajectories, num_points, _ = data.shape
    
    # Calculate reference latitude and longitude
    ref_lat, ref_lon = np.mean(data[:, 0, 1]), np.mean(data[:, 0, 0])
    print(f"Reference Latitude: {ref_lat:.6f}")
    print(f"Reference Longitude: {ref_lon:.6f}")
    
    # Convert lat/lon to meters
    data_meters = np.zeros((num_trajectories, num_points, 2))
    for i in range(num_trajectories):
        data_meters[i, :, 0], data_meters[i, :, 1] = lat_lon_to_meters(data[i, :, 1], data[i, :, 0], ref_lat, ref_lon)
    
    # Calculate statistics for each time stamp
    avg_trajectory = np.mean(data_meters, axis=0)
    std_trajectory = np.std(data_meters, axis=0)
    
    # Percentile-based selection
    lower_bound_percentile = np.percentile(data_meters, 2.5, axis=0)
    upper_bound_percentile = np.percentile(data_meters, 97.5, axis=0)
    
    # 3-standard-deviation selection
    lower_bound_std = avg_trajectory - 3 * std_trajectory
    upper_bound_std = avg_trajectory + 3 * std_trajectory
    
    # Select trajectories closest to the bounds
    def find_closest_trajectory(trajectories, target):
        distances = np.sum(np.sqrt(np.sum((trajectories - target)**2, axis=2)), axis=1)
        return trajectories[np.argmin(distances)]
    
    lower_trajectory_percentile = find_closest_trajectory(data_meters, lower_bound_percentile)
    upper_trajectory_percentile = find_closest_trajectory(data_meters, upper_bound_percentile)
    lower_trajectory_std = find_closest_trajectory(data_meters, lower_bound_std)
    upper_trajectory_std = find_closest_trajectory(data_meters, upper_bound_std)
    
    # Print statistics for the first and last time stamps
    for i, label in enumerate(["First", "Last"]):
        idx = 0 if label == "First" else -1
        print(f"\n{label} time stamp statistics:")
        print(f"X coordinate (meters):")
        print(f"  Mean: {avg_trajectory[idx, 0]:.2f}")
        print(f"  Standard deviation: {std_trajectory[idx, 0]:.2f}")
        print(f"  95% Range: ({lower_bound_percentile[idx, 0]:.2f}, {upper_bound_percentile[idx, 0]:.2f})")
        print(f"  3-std Range: ({lower_bound_std[idx, 0]:.2f}, {upper_bound_std[idx, 0]:.2f})")
        print(f"Y coordinate (meters):")
        print(f"  Mean: {avg_trajectory[idx, 1]:.2f}")
        print(f"  Standard deviation: {std_trajectory[idx, 1]:.2f}")
        print(f"  95% Range: ({lower_bound_percentile[idx, 1]:.2f}, {upper_bound_percentile[idx, 1]:.2f})")
        print(f"  3-std Range: ({lower_bound_std[idx, 1]:.2f}, {upper_bound_std[idx, 1]:.2f})")
    
    # Plot trajectories with average and bounds
    plt.figure(figsize=(12, 8))
    for i in range(num_trajectories):
        plt.plot(data_meters[i, :, 0], data_meters[i, :, 1], alpha=0.1, color='gray')
    
    plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(upper_trajectory_percentile[:, 0], upper_trajectory_percentile[:, 1], color='red', linestyle='--', label='97.5th Percentile')
    plt.plot(lower_trajectory_percentile[:, 0], lower_trajectory_percentile[:, 1], color='green', linestyle='--', label='2.5th Percentile')
    plt.plot(upper_trajectory_std[:, 0], upper_trajectory_std[:, 1], color='magenta', linestyle=':', label='+3 Std Dev')
    plt.plot(lower_trajectory_std[:, 0], lower_trajectory_std[:, 1], color='cyan', linestyle=':', label='-3 Std Dev')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Trajectory Map with Average and Bound Trajectories')
    plt.legend()
    plt.savefig('trajectory_map_with_3bound_trajectories.png')
    plt.close()
    
    # Zoomed-in view
    plt.figure(figsize=(12, 8))
    plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='blue', linewidth=2, label='Average Trajectory')
    plt.plot(upper_trajectory_percentile[:, 0], upper_trajectory_percentile[:, 1], color='red', linestyle='--', label='97.5th Percentile')
    plt.plot(lower_trajectory_percentile[:, 0], lower_trajectory_percentile[:, 1], color='green', linestyle='--', label='2.5th Percentile')
    plt.plot(upper_trajectory_std[:, 0], upper_trajectory_std[:, 1], color='magenta', linestyle=':', label='+3 Std Dev')
    plt.plot(lower_trajectory_std[:, 0], lower_trajectory_std[:, 1], color='cyan', linestyle=':', label='-3 Std Dev')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Zoomed-in View of Trajectories with Bounds')
    plt.legend()
    
    # Adjust the plot limits to zoom in
    x_range = np.max(avg_trajectory[:, 0]) - np.min(avg_trajectory[:, 0])
    y_range = np.max(avg_trajectory[:, 1]) - np.min(avg_trajectory[:, 1])
    plt.xlim(np.min(avg_trajectory[:, 0]) - 0.1*x_range, np.max(avg_trajectory[:, 0]) + 0.1*x_range)
    plt.ylim(np.min(avg_trajectory[:, 1]) - 0.1*y_range, np.max(avg_trajectory[:, 1]) + 0.1*y_range)
    
    plt.savefig('zoomed_trajectory_with_3bound_trajectories.png')
    plt.close()

    return avg_trajectory, lower_trajectory_percentile, upper_trajectory_percentile, lower_trajectory_std, upper_trajectory_std

# Run the analysis