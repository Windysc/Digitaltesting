import numpy as np
from scipy.stats import entropy
import os

def calculate_segment_distances(trajectory):
    """Calculate distances between consecutive points in the trajectory."""
    return np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))

def calculate_total_distance(trajectory):
    """Calculate the total distance of the trajectory."""
    return np.sum(calculate_segment_distances(trajectory))

def jsd(p, q):
    """Calculate Jensen-Shannon Divergence between two distributions."""
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def calculate_jsd(ref_dist, comp_dist, num_bins=100):
    """Calculate JSD between reference and comparison distributions."""
    min_val = min(np.min(ref_dist), np.min(comp_dist))
    max_val = max(np.max(ref_dist), np.max(comp_dist))
    bins = np.linspace(min_val, max_val, num_bins)
    
    ref_hist, _ = np.histogram(ref_dist, bins=bins, density=True)
    comp_hist, _ = np.histogram(comp_dist, bins=bins, density=True)
    
    return jsd(ref_hist, comp_hist)

def load_trajectory_data(file_path):
    """Load trajectory data from a numpy file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] == 2:  
                print(f"Successfully loaded trajectory from {file_path}")
                print(f"Trajectory shape: {data.shape}")
                return data
            else:
                print(f"Invalid data shape in {file_path}. Expected (n, m, 2), got {data.shape}")
        else:
            print(f"Unexpected data type in {file_path}. Expected numpy array, got {type(data)}")
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
    return None

def example_usage(data_folder, reference_file):
    print(f"Searching for trajectory data in folder: {data_folder}")
    
    # Load reference trajectory
    ref_trajectory = load_trajectory_data(reference_file)
    if ref_trajectory is None:
        print("Failed to load reference trajectory. Exiting.")
        return

    # Calculate reference metrics
    ref_total_distances = [calculate_total_distance(traj) for traj in ref_trajectory]
    ref_segment_distances = np.concatenate([calculate_segment_distances(traj) for traj in ref_trajectory])
    
    trajectories = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(data_folder, filename)
            trajectory = load_trajectory_data(file_path)
            if trajectory is not None:
                trajectories.append((filename, trajectory))
            else:
                print(f"Failed to load trajectory from {filename}")
    
    if not trajectories:
        print("No valid trajectory data found in the specified folder.")
        return

    print(f"Loaded {len(trajectories)} valid trajectories")

    for filename, trajectory in trajectories:
        print(f"\nProcessing: {filename}")
        
        total_distances = [calculate_total_distance(traj) for traj in trajectory]
        segment_distances = np.concatenate([calculate_segment_distances(traj) for traj in trajectory])

        # Calculate JSD for total distances and segment distances
        jsd_td = calculate_jsd(ref_total_distances, total_distances)
        jsd_sd = calculate_jsd(ref_segment_distances, segment_distances)

        print(f"Results for {filename}:")
        print(f"JSD-TD (Total Distance): {jsd_td}")
        print(f"JSD-SD (Segment Distance): {jsd_sd}")

        print("Summary Statistics:")
        print(f"Generated Total Distance: Mean = {np.mean(total_distances)}, Std = {np.std(total_distances)}")
        print(f"Generated Segment Distance: Mean = {np.mean(segment_distances)}, Std = {np.std(segment_distances)}")

    # Print reference statistics once at the end
    print("\nReference Statistics:")
    print(f"Reference Total Distance: Mean = {np.mean(ref_total_distances)}, Std = {np.std(ref_total_distances)}")
    print(f"Reference Segment Distance: Mean = {np.mean(ref_segment_distances)}, Std = {np.std(ref_segment_distances)}")

if __name__ == "__main__":
    data_folder = "/home/junze/.jupyter/Evaluation"
    reference_file = "/home/junze/.jupyter/data_train_766.npy"  # Specify the path to your reference file
    example_usage(data_folder, reference_file)