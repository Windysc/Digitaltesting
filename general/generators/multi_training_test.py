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
from typing import Dict, List, Tuple


dataset = np.load(r"/home/junze/exp1/data_train_766.npy")
architecture = tsgm.models.architectures.zoo["vae_conv5"](91, 2, 10)
encoder, decoder = architecture.encoder, architecture.decoder

scaler = tsgm.utils.TSFeatureWiseScaler()
scaled_data = scaler.fit_transform(dataset)

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
        
    def on_train_end(self, logs=None):
        final_samples = self.model.generate(self.generate_samples)
        self.generated_data['final'] = final_samples

def train_vae(scaled_data: np.ndarray, encoder: keras.Model, decoder: keras.Model, 
              epochs: int = 1000, batch_size: int = 32, 
              beta: float = 1.0) -> Tuple[tsgm.models.cvae.BetaVAE, VAEMonitor, keras.callbacks.History]:
    vae = tsgm.models.cvae.BetaVAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam())
    
    monitor = VAEMonitor()
    
    history = vae.fit(
        scaled_data, 
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=[monitor]
    )
    
    return vae, monitor, history

def multi_train_vae(scaled_data: np.ndarray, encoder: keras.Model, decoder: keras.Model, 
                    configs: List[Dict]) -> Dict[str, Tuple[tsgm.models.cvae.BetaVAE, VAEMonitor, keras.callbacks.History]]:
    results = {}
    for i, config in enumerate(configs):
        print(f"Training configuration {i+1}/{len(configs)}")
        vae, monitor, history = train_vae(scaled_data, encoder, decoder, **config)
        results[f"config_{i+1}"] = (vae, monitor, history)
    return results

def plot_multiple_losses(results: Dict[str, Tuple[tsgm.models.cvae.BetaVAE, VAEMonitor, keras.callbacks.History]], 
                         save_path: str = 'vae_training_losses_comparison.png'):
    plt.figure(figsize=(12, 8))
    for config_name, (_, monitor, _) in results.items():
        plt.plot(monitor.losses, label=config_name)
    plt.title('VAE Training Losses Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_generated_data(results: Dict[str, Tuple[tsgm.models.cvae.BetaVAE, VAEMonitor, keras.callbacks.History]], 
                        save_path: str = 'generated_data.npz'):
    save_dict = {}
    for config_name, (_, monitor, _) in results.items():
        save_dict[config_name] = monitor.generated_data
    np.savez(save_path, **save_dict)
    
configs = [
    {"epochs": 1000, "batch_size": 64, "beta": 1.0},
    {"epochs": 2000, "batch_size": 64, "beta": 1.0},
    {"epochs": 3000, "batch_size": 64, "beta": 1.0},
    {"epochs": 2000, "batch_size": 64, "beta": 1.5},
    {"epochs": 3000, "batch_size": 64, "beta": 1.5},
]

results = multi_train_vae(scaled_data, encoder, decoder, configs)
plot_multiple_losses(results)
save_generated_data(results)
