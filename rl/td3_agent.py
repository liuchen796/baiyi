import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, env_dim: int = 20, robot_dim: int = 4):
        super().__init__()
        self.env_dim = env_dim
        self.robot_dim = robot_dim
        embed_dim = 32

        self.lidar_embed = nn.Linear(1, embed_dim)
        pe = torch.zeros(env_dim, embed_dim)
        position = torch.arange(0, env_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pos_encoding", pe)

        self.attn_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                     dim_feedforward=64, batch_first=True)

        self.robot_layer = nn.Linear(robot_dim, 64)
        self.fc1 = nn.Linear(embed_dim * env_dim + 64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[:, :self.env_dim].unsqueeze(-1)
        robot = state[:, self.env_dim:]

        env_embed = self.lidar_embed(lidar)
        env_embed = env_embed + self.pos_encoding
        env_attn_out = self.attn_layer(env_embed)

        env_flat = env_attn_out.reshape(env_attn_out.size(0), -1)
        robot_feat = F.relu(self.robot_layer(robot))
        combined = torch.cat([env_flat, robot_feat], dim=1)
        x = F.relu(self.fc1(combined))
        y = self.fc2(x)
        y = F.relu(x + y)
        return self.tanh(self.fc_out(y))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class TD3:
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 env_dim: int = 20, robot_dim: int = 4, buffer_size: int = int(1e6)):
        self.actor = Actor(state_dim, action_dim, env_dim, robot_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, env_dim, robot_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) +
                                                 list(self.critic2.parameters()), lr=1e-3)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.max_action = max_action

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state_tensor)
        return action.cpu().data.numpy().flatten()

    def train(self, iterations: int, batch_size: int = 100, gamma: float = 0.99,
              tau: float = 0.005, policy_noise: float = 0.2, noise_clip: float = 0.5,
              policy_freq: int = 2, uncertainty_threshold: float = 0.5):
        for it in range(iterations):
            states, actions, rewards, dones, next_states, indices, weights = \
                self.replay_buffer.sample_batch(batch_size)

            state = torch.FloatTensor(states).to(device)
            action = torch.FloatTensor(actions).to(device)
            reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            done = torch.FloatTensor(dones).unsqueeze(1).to(device)
            next_state = torch.FloatTensor(next_states).to(device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

            with torch.no_grad():
                noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                target_q1 = self.critic1_target(next_state, next_action)
                target_q2 = self.critic2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * gamma * target_q

            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)
            td_error1 = (current_q1 - target_q).detach()
            td_error2 = (current_q2 - target_q).detach()
            td_errors = (torch.abs(td_error1) + torch.abs(td_error2)) / 2.0
            self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy().flatten())

            loss1 = F.mse_loss(current_q1, target_q, reduction='none')
            loss2 = F.mse_loss(current_q2, target_q, reduction='none')
            critic_loss = (weights * loss1 + weights * loss2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                q_diff = torch.abs(current_q1 - current_q2).mean().item()
                if q_diff < uncertainty_threshold:
                    actor_loss = -self.critic1(state, self.actor(state)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

