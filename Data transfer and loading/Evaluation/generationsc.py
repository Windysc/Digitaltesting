import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)

memory_limit = 4096
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
        )
        print(f"Memory limit set to {memory_limit} MB for GPU.")
    except RuntimeError as e:
        print(e)
        
import tsgm
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tsgm.models.monitors import VAEMonitor
from tsgm.utils import TSFeatureWiseScaler
from tsgm.models.cvae import BetaVAE
from pyproj import Proj
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K

        
tf.config.run_functions_eagerly(True)

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


class InterpretableVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, smoothness_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.smoothness_weight = smoothness_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.smoothness_loss_tracker = keras.metrics.Mean(name="smoothness_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.smoothness_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling(z_mean, z_log_var)
        return self.decoder(z)

    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            smoothness_loss = tf.reduce_mean(tf.square(tf.experimental.numpy.diff(reconstruction, axis=1)))
            total_loss = reconstruction_loss + self.beta * kl_loss + self.smoothness_weight * smoothness_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.smoothness_loss_tracker.update_state(smoothness_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "smoothness_loss": self.smoothness_loss_tracker.result(),
        }

    def generate(self, num_samples):
        z = tf.random.normal(shape=(num_samples, self.encoder.outputs[0].shape[1]))
        return self.decoder(z)

def create_vae_architecture(input_shape, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(32, 3, activation="relu", padding="same")(encoder_inputs)
    x = keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.Conv1D(64, 3, activation="relu", padding="same", strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(64, activation="relu")(latent_inputs)
    x = keras.layers.Dense(input_shape[0] * 32, activation="relu")(x)
    x = keras.layers.Reshape((input_shape[0], 32))(x)
    x = keras.layers.Conv1DTranspose(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.Conv1DTranspose(32, 3, activation="relu", padding="same")(x)
    decoder_outputs = keras.layers.Conv1DTranspose(input_shape[-1], 3, activation="linear", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    return encoder, decoder

# Training process
def train_vae(vae, data, epochs=1000, batch_size=32):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    history = vae.fit(data, epochs=epochs, batch_size=batch_size, callbacks=[callback], verbose=2)
    return history

# Main execution
input_shape = (max_length, 2)  # (time_steps, features)
LATENT_DIM = 32  # Reduced latent dimensions

encoder1, decoder1 = create_vae_architecture(input_shape, LATENT_DIM)
encoder2, decoder2 = create_vae_architecture(input_shape, LATENT_DIM)

vae1 = InterpretableVAE(encoder1, decoder1, beta=0.001, smoothness_weight=0.1)
vae2 = InterpretableVAE(encoder2, decoder2, beta=0.001, smoothness_weight=0.1)

vae1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
vae2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

history1 = train_vae(vae1, scaled_data1)
history2 = train_vae(vae2, scaled_data2)

# Generate samples
num_samples = 20
generated_data1 = vae1.generate(num_samples)
generated_data2 = vae2.generate(num_samples)

# Inverse transform the generated data
generated_data1_unscaled = scaler.inverse_transform(tf.reshape(generated_data1, [-1, 2])).reshape(generated_data1.shape)
generated_data2_unscaled = scaler.inverse_transform(tf.reshape(generated_data2, [-1, 2])).reshape(generated_data2.shape)

# Save generated data
np.save('data_trajectory1.npy', generated_data1_unscaled)
np.save('data_trajectory2.npy', generated_data2_unscaled)

def evaluate_and_visualize(real_data, generated_data, title):
    print(f"\n{title}")
    print("Real data shape:", real_data.shape)
    print("Generated data shape:", generated_data.shape)
    
    # Ensure both arrays have the same shape
    min_length = min(real_data.shape[1], generated_data.shape[1])
    real_data = real_data[:, :min_length, :]
    generated_data = generated_data[:, :min_length, :]
    
    # Compute the average distance
    avg_distance = np.mean(np.sqrt(np.sum((real_data - generated_data)**2, axis=(1, 2))))
    print(f"Average distance between real and generated trajectories: {avg_distance}")
    
    # Visualization
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
    
    # Plot real trajectories
    plt.plot(real_data1[0, :, 0], real_data1[0, :, 1], 'b-', label='Real Trajectory 1')
    plt.plot(real_data2[0, :, 0], real_data2[0, :, 1], 'g-', label='Real Trajectory 2')
    
    # Plot generated trajectories
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

# Call the function to merge and visualize trajectories
merge_and_visualize_trajectories(dataset1_padded, dataset2_padded, generated_data1_unscaled, generated_data2_unscaled)

# Save generated data
np.save('data_trajectory1.npy', generated_data1_unscaled.numpy())
np.save('data_trajectory2.npy', generated_data2_unscaled.numpy())

print("Evaluation complete. Plots saved in 'evaluation_plots' directory.")

# Plot loss history
plt.figure(figsize=(12, 6))
plt.plot(history1.history['loss'], label='Trajectory 1 Loss')
plt.plot(history2.history['loss'], label='Trajectory 2 Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')  # Use log scale for better visualization
plt.savefig('evaluation_plots/training_loss.png')
plt.close()