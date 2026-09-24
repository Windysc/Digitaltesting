"""
PPO.py -- the `PPO` class that main_attack_ppo_enc.py and main_attack_ppo_scen.py import.

Rebuilt 2026-09-15.  Same constructor and methods as the public PPO-PyTorch
reference the scripts were written against (taken from the ppo_agent.py of the
earlier Ship_envre_v2 build, which left the repository on 2026-09-24):

    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                eps_clip, has_continuous_action_space, action_std_init)
    a = agent.select_action(state)          # stores state/action/logprob/value in agent.buffer
    agent.buffer.rewards.append(r); agent.buffer.is_terminals.append(done)
    agent.update()                          # PPO update on the buffer, then clears it
    agent.decay_action_std(rate, min_std)   # continuous only
    agent.save(path); agent.load(path)

Device: CPU by default (a 64-unit MLP trains faster on CPU than on the
RTX 4060 here); set PPO_DEVICE=cuda to use the GPU.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

_requested = os.environ.get('PPO_DEVICE', 'cpu').lower()
device = torch.device('cuda' if _requested.startswith('cuda') and torch.cuda.is_available() else 'cpu')


class RolloutBuffer:
    def __init__(self):
        self.actions, self.states, self.logprobs = [], [], []
        self.rewards, self.state_values, self.is_terminals = [], [], []

    def clear(self):
        self.__init__()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, hidden=64):
        super().__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init ** 2).to(device)
            self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, action_dim), nn.Tanh())
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std ** 2).to(device)

    def act(self, state, deterministic=False):
        if self.has_continuous_action_space:
            mean = self.actor(state)
            dist = MultivariateNormal(mean, torch.diag(self.action_var).unsqueeze(0))
            action = mean if deterministic else dist.sample()
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        return action.detach(), dist.log_prob(action).detach(), self.critic(state).detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            mean = self.actor(state)
            cov = torch.diag_embed(self.action_var.expand_as(mean))
            dist = MultivariateNormal(mean, cov)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            dist = Categorical(self.actor(state))
        return dist.log_prob(action), self.critic(state), dist.entropy()


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, entropy_coef=0.01,
                 minibatch_size=None, max_grad_norm=0.5):
        self.has_continuous_action_space = has_continuous_action_space
        self.action_std = action_std_init
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.entropy_coef, self.minibatch_size, self.max_grad_norm = entropy_coef, minibatch_size, max_grad_norm
        self.buffer = RolloutBuffer()
        print('PPO agent on device:', device)
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}])
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    # -------------------------------------------------- exploration control
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = max(round(self.action_std - action_std_decay_rate, 4), min_action_std)
            self.set_action_std(self.action_std)

    # ----------------------------------------------------------- acting
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device).unsqueeze(0)
            action, logprob, value = self.policy_old.act(state, deterministic)
        if not deterministic:
            self.buffer.states.append(state.squeeze(0))
            self.buffer.actions.append(action.squeeze(0))
            self.buffer.logprobs.append(logprob.squeeze(0))
            self.buffer.state_values.append(value.squeeze(0))
        if self.has_continuous_action_space:
            return action.squeeze(0).cpu().numpy()
        return int(action.item())

    # ----------------------------------------------------------- learning
    def update(self):
        if len(self.buffer.rewards) == 0:
            return {}
        rewards, discounted = [], 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        n = min(len(rewards), len(self.buffer.states))     # guard against a half-stored last step
        old_states = torch.stack(self.buffer.states[:n]).detach()
        old_actions = torch.stack(self.buffer.actions[:n]).detach()
        old_logprobs = torch.stack(self.buffer.logprobs[:n]).detach()
        old_values = torch.stack(self.buffer.state_values[:n]).detach().squeeze(-1)
        rewards = rewards[:n]
        advantages = (rewards - old_values).detach()

        mb = self.minibatch_size or n
        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        for _ in range(self.K_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                logprobs, values, entropy = self.policy.evaluate(old_states[idx], old_actions[idx])
                values = values.squeeze(-1)
                ratios = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(values, rewards[idx])
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return stats

    # ------------------------------------------------------------- io
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(sd)
        self.policy.load_state_dict(sd)


__all__ = ['PPO', 'RolloutBuffer', 'ActorCritic', 'device']
