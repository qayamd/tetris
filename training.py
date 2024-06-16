import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from environment import Tetris
import random

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def normalize_state(state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return state_tensor / torch.max(state_tensor)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, error, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((*args,))
        else:
            self.memory[self.position] = (*args,)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        batch_priorities = np.asarray(batch_priorities).flatten()
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

def train_dqn():
    # Parameters
    batch_size = 512
    lr = 1e-3
    gamma = 0.99
    initial_epsilon = 1
    final_epsilon = 1e-3
    num_decay_epochs = 8000 
    num_epochs = 20000 
    save_interval = 1000
    replay_memory_size = 30000

    env = Tetris()
    model = DeepQNetwork()
    target_model = DeepQNetwork()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    replay_memory = PrioritizedReplayBuffer(replay_memory_size)
    epsilon = initial_epsilon
    epsilon_min = final_epsilon
    epsilon_decay = (initial_epsilon - final_epsilon) / num_decay_epochs
    target_update_freq = 10
    beta_start = 0.4
    beta_frames = 1000
    max_grad_norm = 10

    rewards = []
    losses = []

    os.makedirs('models', exist_ok=True)

    for episode in range(num_epochs):
        state = env.reset()
        state = normalize_state(state.flatten()).clone().detach()
        total_reward = 0
        done = False
        while not done:
            possible_positions = env.get_possible_positions()
            if len(possible_positions) == 0:
                done = True
                reward = -100
                replay_memory.push(1.0, state, -1, reward, state, done, 0)
                total_reward += reward
                continue
            if random.random() < epsilon:
                action_index = random.randint(0, len(possible_positions) - 1)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action_index = torch.argmax(q_values[:len(possible_positions)]).item()
            action = possible_positions[action_index]
            next_state, reward, done = env.step(action)
            next_state = normalize_state(next_state.flatten()).clone().detach()
            replay_memory.push(1.0, state, action_index, reward, next_state, done, len(possible_positions))
            state = next_state
            total_reward += reward

            if len(replay_memory.memory) >= batch_size:
                beta = beta_start + episode * (1.0 - beta_start) / beta_frames
                batch, indices, weights = replay_memory.sample(batch_size, beta)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, valid_actions_batch = zip(*batch)

                state_batch = torch.stack([s for s in state_batch])
                action_batch = torch.tensor(action_batch)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                next_state_batch = torch.stack([s for s in next_state_batch])
                done_batch = torch.tensor(done_batch, dtype=torch.float32)
                valid_actions_batch = torch.tensor(valid_actions_batch)

                q_values = model(state_batch)
                next_q_values = target_model(next_state_batch)

                max_next_q_values = torch.zeros(batch_size)
                for i in range(batch_size):
                    if valid_actions_batch[i] > 0:
                        max_next_q_values[i] = torch.max(next_q_values[i, :valid_actions_batch[i]], dim=0)[0]
                    else:
                        max_next_q_values[i] = 0.0

                target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
                action_indices = torch.clamp(action_batch, min=0, max=q_values.size(1) - 1)
                q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                loss = (torch.tensor(weights, dtype=torch.float32) * criterion(q_values, target_q_values)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                loss_np = loss.detach().numpy()
                if np.isscalar(loss_np):
                    loss_np = np.array([loss_np])
                replay_memory.update_priorities(indices, loss_np)
                losses.append(loss.item())

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        
        if (episode + 1) % save_interval == 0 or episode + 1 == num_epochs:
            save_path = f"models/dqn_model_{episode + 1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode + 1}/{num_epochs} - Total Reward: {total_reward} - Epsilon: {epsilon:.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards')
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Losses')
    plt.show()

if __name__ == "__main__":
    train_dqn()
