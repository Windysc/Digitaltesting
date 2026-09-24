import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)

memory_limit = 10000
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
import keras
import copy
import sklearn
import sklearn.model_selection
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from sklearn.mixture import GaussianMixture
from tsgm.models.monitors import VAEMonitor


VAEMonitor = VAEMonitor(latent_dim=100, output_dim=2, save=True)

path = '/home/junze/.jupyter/Train_VAE_full/datasettest_2.npy'
dataset = np.load(path)
n = os.path.basename(path)

print("Original dataset shape:", dataset.shape)
print("Sample of original data:", dataset[:5])

# Normalize each feature independently
scaled_data = np.zeros_like(dataset)
for i in range(dataset.shape[1]):
    column = dataset[:, i]
    scaled_data[:, i] = (column - np.min(column)) / (np.max(column) - np.min(column))

print("Scaled data shape:", scaled_data.shape)
print("Sample of scaled data:", scaled_data[:5])

input_dim = scaled_data.shape[1]
architecture = tsgm.models.architectures.zoo["vae_conv5"](input_dim, 2, 10)
encoder, decoder = architecture.encoder, architecture.decoder

class VAEMonitor(keras.callbacks.Callback):
    def __init__(self, generate_samples=20, save_interval=100):
        super().__init__()
        self.generate_samples = generate_samples
        self.save_interval = save_interval
        self.losses = []
        self.generated_data = {}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.losses.append(loss)
        
        if (epoch + 1) % self.save_interval == 0 or epoch == 0:
            generated = self.model.generate(self.generate_samples)
            self.generated_data[epoch] = generated
            
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}")
        
    def on_train_end(self, logs=None):
  
        final_samples = self.model.generate(self.generate_samples)
        self.generated_data['final'] = final_samples
        
        np.save('generated_data.npy', self.generated_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('vae_training_loss.png')
        plt.close()

def custom_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    kl_loss = -0.5 * tf.reduce_sum(1 + vae.z_log_var - tf.square(vae.z_mean) - tf.exp(vae.z_log_var), axis=-1)
    temporal_loss = tf.reduce_mean(tf.square(y_true[:, 1:] - y_true[:, :-1] - (y_pred[:, 1:] - y_pred[:, :-1])), axis=-1)
    return reconstruction_loss + 0.1 * kl_loss + 0.5 * temporal_loss

def train_vae(scaled_data, encoder, decoder, epochs=1000, batch_size=64):
    vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=custom_loss)
    
    monitor = VAEMonitor()
    
    print("Input data shape:", scaled_data.shape)
    print("Sample input data:", scaled_data[:5])
    
    history = vae.fit(
        scaled_data, 
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=[monitor]
    )
    
    print("Training complete. Generating sample output...")
    sample_output = vae.generate(5)
    print("Sample output shape:", sample_output.shape)
    print("Sample output:", sample_output)
    
    return vae, monitor, history

vae, monitor, history = train_vae(scaled_data, encoder, decoder)

print("Generating new data...")
data_x = vae.generate(100)

denormalized_data = np.zeros_like(data_x)
for i in range(data_x.shape[1]):
    column = data_x[:, i]
    mean = np.mean(dataset[:, i])
    std = np.std(dataset[:, i])
    denormalized_data[:, i] = column * std + mean

noise = np.random.normal(0, 0.0008, denormalized_data.shape)
denormalized_data += noise


for i in range(denormalized_data.shape[1]):
    min_val = np.min(dataset[:, i])
    max_val = np.max(dataset[:, i])
    denormalized_data[:, i] = np.clip(denormalized_data[:, i], min_val, max_val)
print("Generated data shape:", denormalized_data.shape)
print("Sample of generated data:", denormalized_data[:5])

np.save('/home/junze/.jupyter/Train_VAE_full/data_f2x', denormalized_data)
print("Data saved successfully.")
