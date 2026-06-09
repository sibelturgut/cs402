import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from panda_env_force_keyboard import PandaEnv

def train_behavioral_cloning(epochs=150, batch_size=512):
    print("--- Loading Human Dataset ---")
    try:
        obs_data = np.load("dataset/human_obs.npy")
        action_data = np.load("dataset/human_actions.npy")
        
        # 1. Find the frames where you actually pressed W or S
        active_indices = np.where(np.abs(action_data[:, 0]) > 0.01)[0]
        
        # 2. Find the frames where you were doing nothing
        idle_indices = np.where(np.abs(action_data[:, 0]) <= 0.01)[0]
        
        # 3. Keep ALL active frames, but only keep 5% of the idle frames
        np.random.shuffle(idle_indices)
        idle_indices_kept = idle_indices[:int(len(idle_indices) * 0.05)]
        
        # 4. Combine them together and shuffle
        balanced_indices = np.concatenate([active_indices, idle_indices_kept])
        np.random.shuffle(balanced_indices)
        
        obs_data = obs_data[balanced_indices]
        action_data = action_data[balanced_indices]
        print(f"Loaded {len(obs_data)} transitions.")
    except FileNotFoundError:
        print("Error: Dataset not found. Run record_human_data.py first!")
        return

    obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
    action_tensor = torch.tensor(action_data, dtype=torch.float32)

    dataset = TensorDataset(obs_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    env = PandaEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    
    # Initialize PPO only to build the brain architecture
    model = PPO("MlpPolicy", env, verbose=0)

    print("--- Starting Behavioral Cloning ---")
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    model.policy.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(model.device)
            batch_actions = batch_actions.to(model.device)

            # Get AI's predicted distribution and calculate error against human action
            distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_actions)
            loss = -log_prob.mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.4f}")

    # Save the cloned model
    os.makedirs("logs/force", exist_ok=True)
    save_path = "logs/force/pure_bc_model.zip"
    model.save(save_path)
    print(f"\n--- Training Complete! Model saved to {save_path} ---")

if __name__ == '__main__':
    train_behavioral_cloning(epochs=150)