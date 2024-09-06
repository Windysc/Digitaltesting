import pandas as pd
import numpy as np
import tsgm
import os 
from scipy import interpolate

def interpolation_downsample(data, num_points):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    elif len(data.shape) > 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")
    
    original_points = np.arange(len(data))
    new_points = np.linspace(0, len(data) - 1, num_points)
    
    if data.shape[1] == 1:
        tck = interpolate.splrep(original_points, data[:, 0], s=0)
        return interpolate.splev(new_points, tck)
    else:
        tck, _ = interpolate.splprep([data[:, i] for i in range(data.shape[1])], u=original_points, s=0)
        return np.column_stack(interpolate.splev(new_points, tck))


path = '/home/junze/.jupyter/head-on/1/2.csv'

df = pd.read_csv(path)

x = os.path.basename(path)

lat_lon = df[['longitude_degrees', 'latitude_degrees']].values
speed = df[['speed']].values
downsampled_data = interpolation_downsample(lat_lon, num_points=100)


np.save('trajectory_data.npy', lat_lon+speed)


dataset = downsampled_data.reshape(1, 100, 2)

aug_model = tsgm.models.augmentations.GaussianNoise()
samples = aug_model.generate(X=dataset, n_samples=500, variance=0.0001)
np.save(rf"/home/junze/.jupyter/Train_VAE_full/dataset10-4.npy", samples)

print(np.shape(samples))