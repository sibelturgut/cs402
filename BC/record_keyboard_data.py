import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import raisimpy as raisim
from panda_env_force_keyboard import PandaEnv
from keyboard_controller import KeyboardController
# from joystick_controller import JoystickController

def record_data(episodes=15):
    env = PandaEnv()
    env = TimeLimit(env, max_episode_steps=5000)
    
    server = raisim.RaisimServer(env.unwrapped.world)
    server.launchServer(8080)
    
    keyboard = KeyboardController()
    # keyboard = JoystickController()
    
    recorded_obs = []
    recorded_actions = []
    
    print("--- Starting Human Data Collection ---")
    print(f"Goal: Drive the robot for {episodes} episodes.")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        print(f"\nEpisode {ep+1}/{episodes} started. Drive!")
        
        while True:
            # 1. Calculate the current force magnitude from the observation
            # obs[-5:-2] grabs the x, y, and z forces.
            current_force_vec = obs[-5:-2] 
            force_mag = np.linalg.norm(current_force_vec)
            
            # 2. Pass it to the controller to update the UI
            action = keyboard.get_action(current_force_mag=force_mag)
            
            # Save the state and action
            recorded_obs.append(obs)
            recorded_actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                print(f"Episode {ep+1} finished. Total Reward: {episode_reward:.2f}")
                break

    # Save to disk
    os.makedirs("dataset", exist_ok=True)
    np.save("dataset/human_obs.npy", np.array(recorded_obs))
    np.save("dataset/human_actions.npy", np.array(recorded_actions))
    
    print(f"\n--- Saved {len(recorded_obs)} frames of data to /dataset! ---")
    server.killServer()

if __name__ == '__main__':
    record_data(episodes=15)