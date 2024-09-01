import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Proj
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)

    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("Memory limit set to 4096 MB for GPU.")
    except RuntimeError as e:
        print(e)

# Utility functions
def latlon_to_xy(lat, lon):
    proj = Proj(proj='utm', zone=30, ellps='WGS84', preserve_units=False)
    x, y = proj(lon, lat)
    return x, y

def read_trajectory_data(csv_path):
    df = pd.read_csv(csv_path)
    x, y = latlon_to_xy(df['latitude_degrees'].values, df['longitude_degrees'].values)
    trajectory_data = np.column_stack((x, y))
    return trajectory_data.reshape(1, *trajectory_data.shape)

# Load datasets
dataset1 = read_trajectory_data("/home/junze/.jupyter/head-on/1/1.csv")
dataset2 = read_trajectory_data("/home/junze/.jupyter/head-on/1/2.csv")

# Determine the maximum length of the datasets
max_length = max(dataset1.shape[1], dataset2.shape[1])

def pad_dataset(dataset, max_length):
    pad_length = max_length - dataset.shape[1]
    if pad_length > 0:
        padding = np.zeros((1, pad_length, dataset.shape[2]))
        return np.concatenate((dataset, padding), axis=1)
    return dataset

dataset1_padded = pad_dataset(dataset1, max_length)
dataset2_padded = pad_dataset(dataset2, max_length)

# Scale data
scaler = MinMaxScaler()
scaled_data1 = scaler.fit_transform(dataset1_padded.reshape(-1, 2)).reshape(dataset1_padded.shape)
scaled_data2 = scaler.transform(dataset2_padded.reshape(-1, 2)).reshape(dataset2_padded.shape)

# VAE model
latent_dim = 32

class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape)
        self.decoder = self.build_decoder(input_shape)

    def build_encoder(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self, input_shape):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation="relu")(latent_inputs)
        x = layers.Dense(input_shape[0] * 64, activation="relu")(x)
        x = layers.Reshape((input_shape[0], 64))(x)
        x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        outputs = layers.Conv1D(2, 3, activation="linear", padding="same")(x)
        return keras.Model(latent_inputs, outputs, name="decoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

# Create and train VAE models
vae1 = VAE((max_length, 2), latent_dim)
vae2 = VAE((max_length, 2), latent_dim)

vae1.compile(optimizer=keras.optimizers.Adam())
vae2.compile(optimizer=keras.optimizers.Adam())

# Ensure the data has the correct shape
scaled_data1 = scaled_data1.reshape(1, *scaled_data1.shape[1:])
scaled_data2 = scaled_data2.reshape(1, *scaled_data2.shape[1:])

history1 = vae1.fit(scaled_data1, epochs=1000, batch_size=1, verbose=1)
history2 = vae2.fit(scaled_data2, epochs=1000, batch_size=1, verbose=1)

# Generate samples
num_samples = 20

def generate_samples(vae, num_samples):
    z = tf.random.normal(shape=(num_samples, latent_dim))
    return vae.decoder(z)

generated_data1 = generate_samples(vae1, num_samples)
generated_data2 = generate_samples(vae2, num_samples)

# Inverse transform the generated data
generated_data1_unscaled = scaler.inverse_transform(generated_data1.numpy().reshape(-1, 2)).reshape(generated_data1.shape)
generated_data2_unscaled = scaler.inverse_transform(generated_data2.numpy().reshape(-1, 2)).reshape(generated_data2.shape)

# Save generated data
np.save('data_trajectory1.npy', generated_data1_unscaled)
np.save('data_trajectory2.npy', generated_data2_unscaled)

# Evaluation and visualization functions
def evaluate_and_visualize(real_data, generated_data, title):
    print(f"\n{title}")
    print("Real data shape:", real_data.shape)
    print("Generated data shape:", generated_data.shape)
    
    min_length = min(real_data.shape[1], generated_data.shape[1])
    real_data = real_data[:, :min_length, :]
    generated_data = generated_data[:, :min_length, :]
    
    avg_distance = np.mean(np.sqrt(np.sum((real_data - generated_data)**2, axis=(1, 2))))
    print(f"Average distance between real and generated trajectories: {avg_distance}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(real_data[0, :, 0], real_data[0, :, 1], 'b-', label='Real')
    ax1.plot(generated_data[0, :, 0], generated_data[0, :, 1], 'r--', label='Generated')
    ax1.set_title(f"{title} - Trajectory Comparison")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    
    ax2.scatter(real_data[0, :, 0], real_data[0, :, 1], c='blue', alpha=0.5, label='Real')
    ax2.scatter(generated_data[0, :, 0], generated_data[0, :, 1], c='red', alpha=0.5, label='Generated')
    ax2.set_title(f"{title} - Point Distribution")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'evaluation_plots/{title.replace(" ", "_")}.png')
    plt.close()

def merge_and_visualize_trajectories(real_data1, real_data2, generated_data1, generated_data2):
    plt.figure(figsize=(12, 8))
    
    plt.plot(real_data1[0, :, 0], real_data1[0, :, 1], 'b-', label='Real Trajectory 1')
    plt.plot(real_data2[0, :, 0], real_data2[0, :, 1], 'g-', label='Real Trajectory 2')
    plt.plot(generated_data1[0, :, 0], generated_data1[0, :, 1], 'r--', label='Generated Trajectory 1')
    plt.plot(generated_data2[0, :, 0], generated_data2[0, :, 1], 'm--', label='Generated Trajectory 2')
    
    plt.title("Merged Trajectories Comparison")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('evaluation_plots/merged_trajectories.png')
    plt.close()

# Evaluate and visualize generated samples
evaluate_and_visualize(dataset1_padded, generated_data1_unscaled, "Trajectory 1")
evaluate_and_visualize(dataset2_padded, generated_data2_unscaled, "Trajectory 2")

# Merge and visualize trajectories
merge_and_visualize_trajectories(dataset1_padded, dataset2_padded, generated_data1_unscaled, generated_data2_unscaled)

print("Evaluation complete. Plots saved in 'evaluation_plots' directory.")

# Plot loss history
plt.figure(figsize=(12, 6))
plt.plot(history1.history['loss'], label='Trajectory 1 Loss')
plt.plot(history2.history['loss'], label='Trajectory 2 Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.savefig('evaluation_plots/training_loss.png')
plt.close()

print("Script execution completed.")