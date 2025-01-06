# Digitaltesting
The documentation for the digital testing of MASS and modifications, based on the ShipAI basis. 

The project aim to create a list workflow of possible agents for training the intentional scenarios of safety risk in maritime series, the task is mainly distributed into two parts.
First is the follow-up mode of the trace, which the adversarial scenario get into the starting point and move up according to the generated and existing route data.
Second is the free chase mode, when the two ship reach the selected area the freedom is given and the catch-up race is then assured.

# Menu
-[The data generation and evaluation part](#The data generation and evaluation part)
-[The Training and simulation part](#The Training and simulation part)
-[Requirements](Requirements)
-[Indications on the existing files:](#Indications on the existing files:) 



## The data generation and evaluation part
The data is not included in the files for the generation as concluding sensitive data, generation models are used in the variance from TSGM built-in models and self-built models based on different networks, which are listed in the train_() files  
Evaluation and transition of the data to the scenario in the input are listed in the evaluation and abstraction documents.
For example, if you want to test the VAE generation network and it's relating training effects:
choose the >Train_VAE_full folder

## The Training and simulation part 
The simluation and the training for the part are referred to the simulator in the ShipAI and Stable-baselines 3 as baseline for the structure, and the files are in the Shipenv_re document
The main simulation file is based on the ppo_sb3_rl.py file and it has two main modes for the training.

If selecting the train mode, the .py file will be working on the sb3 training settings conducted in the file itself and after training will save the agent file after running.

If selecting the eval mode, the .py file will read the agent package and work for 10 episodes on the default settings in the testing setting in the Ship_env.py file and the evalation saving files are mainly two parts: a json file which conclude all the data in the monitored function array in the main training file upwards in the run_and_evaluate function, and pictures revealing the outcome of reward, action change with the step and the trajectory outcome in meters.

## Requirements
### Python interpreter 
The data generation and transition period are based on the Python 3.10 environment,
The ShipAI and relating training period are based on the Python 3.7 environment,
### Packages installing
The generation part are completely runnable in the defalut TSGM requirements, and the second part requirements are printed in the repo for serving needs


## Indications on the existing files:
The files are generally packed in five folders: The three train_vae_full, train_nGAN_full and train_Diff_full folders, the evaluation folder and the Ship_envre folder. For the containing files in the folders, the first three train folders contain the training scripts for the data generation part, which are classified using the network type. To note that, the nGAN folder contains both the cGAN and wGAN training attempt scripts and the Diff folder containing the Diffusion, TrajDiffusion and Unet training attempt scripts but not all the scripts are availiable for use. The availbility for the generation is listed with a(u) in the name of the training script which may be in the form of .py files or jupyter notebook files.

For the evaluation folder the containing three .py files are acting as transfer role for the data to become inportable trace, the jsd-matrix file as the judge for the generation outcome, plot_training for the generation visualization and the scenario_creation for the trace generation.

For the Ship_envre folder things are served mainly for the critical scenario generation, with the ppo_sb3_rl.py as the main file for running and the Ship_env.py and the simulator.py acting as the background settings and the ship simulatioon container. 

The Ship_env.py file is based on the ShipAI project which could be found on the github codespace. The overall settings are in the default style of the Ship_env.py file in the same Gym environment and share the same variables in the scope of controlling. The Ship_env.py in the current file is a completely rebuild and have other factors like border reading, scope calculation, and reward stage comfirmation. For the running of the whole project in test of different scenario occurance, the focus should be mainly on the selection of the boudaries of the variables involved, reading of the data and the policy of the learning and reward.

## Steps and instructions 
For the whole process details relating on the training and generating coding practices:

First select the data for initialization in advance,

For data training usage, just run the ppo_sb3_rl.py with the default settings and you could make adjustments in the latter part of the file for tryouts. All the files conclude the reading of the data file and just change the path of the file in the reading lines, 


selecting the proposed outcome path and change the desired training parameters to cope with the input data to avoid faults.

For the connection between the data training and the scenario generation:



Last for the scenario generation, we use the default Stablebaselines3 agent for the baseline of the model, which could be replaced easily in the following process:


For the scenario creation, the environment settings shall be classified:

For the boundaries of variables:

For the settings of the policies:



The other files are marked with their usage in accordance with the main files.
