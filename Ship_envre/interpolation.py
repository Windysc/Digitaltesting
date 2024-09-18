import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.load('/home/junze/.jupyter/Digitaltesting/Ship_envre/datasettest_2.npy')

trajectory = data[0]

t_original = np.arange(0, len(trajectory) * 9, 9)

interp_lon = interp1d(t_original, trajectory[:, 0], kind='cubic')
interp_lat = interp1d(t_original, trajectory[:, 1], kind='cubic')

t_interpolated = np.arange(0, t_original[-1] + 1)

#do the interpolation
trajectory_interpolated = np.column_stack((
    interp_lon(t_interpolated),
    interp_lat(t_interpolated),
))

#After we do the downsample to cope with the interval of simulator
t_downsampled = np.arange(0, t_original[-1] + 1, 20)
trajectory_downsampled = np.column_stack((
    interp_lon(t_downsampled),
    interp_lat(t_downsampled),
))

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
features = ['Longitude', 'Latitude']

for i in range(2):  
    ax = axs[i]
    ax.plot(t_original, trajectory[:, i], 'ro', label='Original Data')
    ax.plot(t_interpolated, trajectory_interpolated[:, i], 'bo', label='1s Interpolation')
    ax.plot(t_downsampled, trajectory_downsampled[:, i], 'g*', label='20s Downsampled')
    ax.set_title(features[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(features[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('interpolation_results.png')
plt.close()

np.save('downsampled_trajectory2.npy', trajectory_downsampled)

print("Interpolation and downsampling complete. Results saved to 'interpolation_results.png' and 'downsampled_trajectory.npy'.")