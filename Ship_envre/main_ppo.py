import os
import glob
from datetime import datetime
import copy
import torch
import numpy as np

import gym
import os
import math
import csv
import json
import random
import argparse
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import style
# matplotlib.use( 'tkagg' )
style.use("ggplot")
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from time import time
import numpy as np
import pdb

from PPO import PPO
from env_moving_obj import MassTestingEnv, ownship, navigation_target, obstacle


def plot_durations(episode_returns, title='Training...', average_duration=10, show_result=False):
    plt.figure(1)
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    durations_t = torch.tensor(episode_returns, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Episode Rewards')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= average_duration:
        means = durations_t.unfold(0, average_duration, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(average_duration-1), means))
        plt.plot(means.numpy())
        
################################### Training ###################################
def train(args):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = 'MassTestingEnv'

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e7)   # break training loop if timeteps > max_training_timesteps

    print_freq = args.eval_every #max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = args.eval_every #in episode #int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = args.seed         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    ##Step1:own ship
    own_ship = ownship(0, 0, 0, 0, 0, 0)
    ##
    ##Step2:target ship
    ts_list=[]
    ##Step3:obstacle
    ##
    # ob_turn_static=obstacle(id=1, lat=450, long=1300,  sp=0, cog=90,L=400, W=50, Direction=0,risk_range=500)
    # ob_turn_move=obstacle(id=3, lat=450, long=0, sp=8, cog=0,L=200, W=50, Direction=0,risk_range=500)
    # ob_line_static=obstacle(id=2, lat=0, long=1000, sp=0, cog=90,L=400, W=50, Direction=90,risk_range=500)
    # ob_line_move=obstacle(id=2, lat=-1000, long=1000, sp=9, cog=90,L=200, W=50, Direction=90,risk_range=500)
    ##
    
    if args.obst_id == 'line_static':
        ob_list = [ob_line_static]
    if args.obst_id == 'line_move':
        ob_list=[ob_line_move]
    if args.obst_id == 'turn_static':
        ob_list = [ob_turn_static]
    if args.obst_id == 'turn_move':
        ob_list=[ob_turn_move]

    ##Step4:destination
    # nt_line=navigation_target(lat = 0,long = 1750,direction=0,target_deviation_distance=200,target_deviation_direction=15)
    # nt_turn=navigation_target(lat = 750,long = 1500,direction=90,target_deviation_distance=200,target_deviation_direction=15)
    ###################################
    if args.dest_id == 'line':
        nt = nt_line
    if args.dest_id == 'turn':
        nt = nt_turn 
   
    save_dir = args.save_dir
    model_dir=f'{save_dir}/models'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    train_log_dir=f'{save_dir}/train_logs'
    eval_log_dir=f'{save_dir}/eval_logs'
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    # train_sample_dir=f'{save_dir}/train_success_samples'
    # eval_sample_dir=f'{save_dir}/eval_success_samples'
    # os.makedirs(train_sample_dir, exist_ok=True)
    # os.makedirs(eval_sample_dir, exist_ok=True)
    
    with open(f'{save_dir}/args.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)

    eval_id = f'obst_{args.obst_id}_dest_{args.dest_id}'

    print('ob list', ob_list)
    env = MassTestingEnv(own_ship, ts_list, ob_list, nt,
                 duration=args.duration, decision_interval=args.decision_interval,
                 reward_type=args.reward_type, X_LEN=args.map_x_size, Y_LEN=args.map_y_size,
                 save_dir=args.save_dir)
    
    with open(f'{train_log_dir}/train_result.txt', 'w') as f:
        f.write(f'Episode AverageReturn Lens Success \n')

    # with open(f'{train_log_dir}/train_a.txt', 'w') as f:
    #     f.write(f'Episode a_list\n')

    # with open(f'{train_log_dir}/train_action.txt', 'w') as f:
    #     f.write(f'Episode action_list\n')

    eval_env = MassTestingEnv(own_ship, ts_list, ob_list, nt,
                 duration=args.duration, decision_interval=args.decision_interval,
                 reward_type=args.reward_type, X_LEN=args.map_x_size, Y_LEN=args.map_y_size,
                 save_dir=args.save_dir)
    
    
    # eval_envs_list.append(eval_env)
    with open(f'{eval_log_dir}/eval_result.txt', 'w') as f:
        f.write(f'Episode AverageReturn Lens SuccessRate \n')
        
    # with open(f'{eval_log_dir}/eval_a.txt', 'w') as f:
    #     f.write(f'Episode a_list\n')

    # with open(f'{eval_log_dir}/eval_action.txt', 'w') as f:
    #     f.write(f'Episode action_list\n')

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    # log_f_name = train_log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    # print("current logging run number for " + env_name + " : ", run_num)
    # print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    # run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = model_dir #"PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_seed{}.pth".format(env_name, random_seed)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # log_f = open(log_f_name,"w+")
    # log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    train_episode_returns = []
    eval_episode_returns = []
    # steps_done = 0
    # training loop
    best_eval_success = 0.0
    # while time_step <= max_training_timesteps and 
    for i_episode in tqdm(range(1, args.num_episodes+1)):
        # print('Trainng...', time_step)
        state = env.reset()
        current_ep_reward = 0
        train_eps_return = 0
        success_flags = []
        eps_len = 0
        for t in range(1, max_ep_len+1):
            eps_len += 1
            # select action with policy
            action = ppo_agent.select_action(state)
            # state, reward, done, _ = env.step(action)
            state, reward, done, success_flag = env.step(action)
            success_flags.append(success_flag)
            train_eps_return +=  reward

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)




            # if success_flag==1:
            # if 1 in success_flags and -1 not in success_flags:
            #     env.show_path(i_episode)
                # plt.savefig(f'{train_sample_dir}/ship_path_train_episode_{i_episode}.png')
                # with open(f'{train_log_dir}/train_action.txt', 'a') as f:
                    # f.write(f'{i_episode}| '+' '.join(map(str, env.ownship_action)) + '\n')

                # with open(f'{train_log_dir}/train_a.txt', 'a') as f:
                    # f.write(f'{i_episode}| ' +' '.join(map(str, env.ownship_a)) + '\n')
                    
            # break; if the episode is over
            if done:
                # print(success_flags)
                if args.render == 1 and i_episode % 10 == 0:
                    train_episode_returns.append(train_eps_return)
                    plot_durations(train_episode_returns)
                    plt.savefig(f'{save_dir}/train_rl_returns.png')
                    # env.show_scenes()
                    # env.show_parameters()
                    plt.close()
                    # env.show_path(i_episode)
                    # plt.savefig(f'{save_dir}/train_rl_ship_path.png')
                    plt.clf()
                    
                if 1 in success_flags and -1 not in success_flags:
                    success = True
                else:
                    success = False
                    
                with open(f'{train_log_dir}/train_result.txt', 'a') as f:
                    f.write(f'{i_episode} {train_eps_return} {env.destination_step} {success} \n')
                    
                state = env.reset()
                eps_len = 0
                break

        # log in logging file
        # if time_step % log_freq == 0:
        # if i_episode % log_freq == 0:

        #     # log average reward till last episode
        #     log_avg_reward = log_running_reward / log_running_episodes
        #     log_avg_reward = round(log_avg_reward, 4)

        #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
        #     log_f.flush()

        #     log_running_reward = 0
        #     log_running_episodes = 0
            
        # printing average reward
        # if time_step % print_freq == 0:
        if i_episode % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        # if time_step % save_model_freq == 1:
        if i_episode % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        # if time_step % args.eval_every == 0:
        if i_episode % args.eval_every == 0:
            print(f'Evaluating at timestep {time_step}, episode {i_episode}...')
            ppo_agent.save(checkpoint_path.replace('.pth', f'_episode{i_episode}.pth'))
            test_ppo_agent = copy.deepcopy(ppo_agent)
            eval_returns = []
            eval_steps = []
            eval_success = []
            eval_dicts = []
            success_cnt = 0
            for i_eval in tqdm(range(args.num_eval)):
                eval_eps_return, eval_eps_step, eval_eps_success, eval_eps_a, eval_eps_action, eval_eps_dict = \
                    eval(eval_env, test_ppo_agent, max_ep_len)
                eval_returns.append(eval_eps_return)
                eval_steps.append(eval_eps_step)
                eval_success.append(eval_eps_success)
                eval_dicts.append(eval_eps_dict)
                
    
                if eval_eps_success == 1:
                    success_cnt += 1
                    print(f'Reach destination at eval env {eval_id} episode {i_eval}!')
                    if success_cnt < 5:
                        eval_env.show_scenes()
                        plt.close()
                        plt.clf()
                    # plt.savefig(f'{eval_sample_dir}/eval_scene_episode{i_episode}_{i_eval}.png')
                    # with open(f'{eval_log_dir}/eval_action.txt', 'a') as f:
                    #     f.write(f'[{i_episode}|{i_eval}] '+' '.join(map(str, eval_eps_action)) + '\n')

                    # with open(f'{eval_log_dir}/eval_a.txt', 'a') as f:
                    #     f.write(f'[{i_episode}|{i_eval}] ' +' '.join(map(str, eval_eps_a)) + '\n')
                
            with open(f'{eval_log_dir}/eval_result.txt', 'a') as f:
                f.write(f'{i_episode} {np.mean(eval_returns)} {np.mean(eval_steps)} {np.mean(eval_success)} \n')
            
            if np.mean(eval_success) > best_eval_success:
                print('Saving best checkpoint with success rate of:', np.mean(eval_success))
                ppo_agent.save(checkpoint_path.replace('.pth', '_best.pth'))
                best_eval_success = np.mean(eval_success)
                
            with open(f'{eval_log_dir}/eval_dicts_episode_{i_episode}.csv', 'w', encoding='utf8') as fout:
                fc = csv.DictWriter(fout, fieldnames=eval_dicts[0].keys(),)
                fc.writeheader()
                fc.writerows(eval_dicts)
                
            # eval_episode_returns.append(np.mean(eval_returns))
            # plot_durations(eval_episode_returns, title='Evaluating...', average_duration=3)
            # plt.savefig(f'{save_dir}/eval_rl_returns.png')
            

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        
        

    # log_f.close()
    # env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def eval(env, ppo_agent, max_ep_len=1000):
    state = env.reset()
    # current_ep_reward = 0
    eps_return = 0
    success_flags = []
    eps_success = 0
    for t in range(1, max_ep_len+1):
        # select action with policy
        action = ppo_agent.select_action(state)
        # state, reward, done, _ = env.step(action)
        state, reward, done, success_flag = env.step(action)
        success_flags.append(success_flag)
        eps_return +=  reward
        # current_ep_reward += reward
        
        # break; if the episode is over
        if done:
            evaluation_dict = env.evaluation()
            eps_steps = env.destination_step
            eps_a = env.live_ownship.a
            eps_action = env.ownship_action
            
            if 1 in success_flags and -1 not in success_flags:
                eps_success = 1
                
            break
        # env.close()
        # ppo_agent.buffer.clear()
    return eps_return, eps_steps, eps_success, eps_a, eps_action, evaluation_dict

def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default='opensea_Oct2023/logs',
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_start", type=float, default=0.9)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=1000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decision_interval", type=int, default=600)
    parser.add_argument("--eval_every", type=int, default=60)
    parser.add_argument("--num_episodes", type=int, default=6000)
    parser.add_argument("--num_eval", type=int, default=100)
    parser.add_argument("--map_x_size", type=int, default=2000)
    parser.add_argument("--map_y_size", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=60000)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--reward_type", type=str, default='final_step_reward')
    parser.add_argument("--eval_id", default='fixed')
    parser.add_argument("--obst_id", type=str, default='turn2')
    parser.add_argument("--dest_id", type=str, default='turn')
    args = parser.parse_args()

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer
    # BATCH_SIZE = 128
    # GAMMA = 0.99
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 1000
    # TAU = 0.005
    # LR = 3e-4
    return args


if __name__ == '__main__':
    args = gen_args()
    train(args)