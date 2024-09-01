import numpy as np

def load_data(file_path):
    """
    Load GPS data from a .npy file.
    
    :param file_path: Path to the .npy file
    :return: Numpy array of shape (x, y, 2) where x is the number of trajectories,
             y is the number of points per trajectory, and 2 represents (x, y) coordinates
    """
    return np.load(file_path)

def calculate_normalization_params(original_data):
    """
    Calculate normalization parameters from the original dataset.
    
    :param original_data: Numpy array of shape (x, y, 2)
    :return: tuple of (min_x, min_y, max_x, max_y)
    """
    min_x = np.min(original_data[:, :, 0])
    min_y = np.min(original_data[:, :, 1])
    max_x = np.max(original_data[:, :, 0])
    max_y = np.max(original_data[:, :, 1])
    return min_x, min_y, max_x, max_y

def denormalize_data(normalized_data, min_vals, max_vals):
    """
    Denormalize the data back to the original scale.
    
    :param normalized_data: Numpy array of shape (x, y, 2) with normalized values
    :param min_vals: tuple of (min_x, min_y)
    :param max_vals: tuple of (max_x, max_y)
    :return: Numpy array of shape (x, y, 2) with denormalized values
    """
    min_x, min_y = min_vals
    max_x, max_y = max_vals
    
    denormalized = np.zeros_like(normalized_data)
    denormalized[:, :, 0] = normalized_data[:, :, 0] * (max_x - min_x) + min_x
    denormalized[:, :, 1] = normalized_data[:, :, 1] * (max_y - min_y) + min_y
    
    return denormalized

def process_dataset(original_file, normalized_file, output_file):
    """
    Process a single dataset: load original and normalized data,
    calculate parameters, denormalize, and save the result.
    
    :param original_file: path to the original dataset .npy file
    :param normalized_file: path to the normalized dataset .npy file
    :param output_file: path to save the denormalized dataset .npy file
    """
    # Load data
    original_data = load_data(original_file)
    normalized_data = load_data(normalized_file)
    
    # Calculate normalization parameters
    min_x, min_y, max_x, max_y = calculate_normalization_params(original_data)
    
    # Denormalize data
    denormalized_data = denormalize_data(
        normalized_data, 
        (min_x, min_y), 
        (max_x, max_y)
    )
    
    # Save denormalized data
    np.save(output_file, denormalized_data)
    print(f"Denormalized data saved to {output_file}")

def main():
    # Process first dataset
    process_dataset(
        '/home/junze/.jupyter/data_train_766.npy',
        '/home/junze/.jupyter/Evaluation/data_1x.npy',
        'denormalized_dataset1.npy'
    )
    
    # Process second dataset
    process_dataset(
        '/home/junze/.jupyter/data_train_766.npy',
        '/home/junze/.jupyter/Evaluation/data_2x.npy',
        'denormalized_dataset2.npy'
    )

if __name__ == "__main__":
    main()