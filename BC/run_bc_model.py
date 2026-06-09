import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import raisimpy as raisim
from stable_baselines3 import PPO
from panda_env_force_keyboard import PandaEnv

def run_and_graph():
    # 1. Setup Environment
    env = PandaEnv()
    env = TimeLimit(env, max_episode_steps=3000) # 30 seconds
    
    server = raisim.RaisimServer(env.unwrapped.world)
    server.launchServer(8080)
    
    model_path = "logs/force/pure_bc_model.zip"
    if not os.path.exists(model_path):
        print("Model not found! Run train_bc.py first.")
        return

    print("--- Loading Pure BC Model ---")
    model = PPO.load(model_path, env=env)
    print("Running 1 evaluation episode. Watch the viewer...")
    
    # 2. Data tracking lists
    history_steps = []
    history_forces = []
    
    obs, _ = env.reset()
    
    # 3. Run exactly ONE episode
    for step in range(3000):
        # deterministic=True ensures strict copying of your behavior
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Extract the current force magnitude from the observation
        # (Assuming force vector is at obs[-5:-2] based on previous setup)
        current_force_vec = obs[-5:-2]
        force_mag = np.linalg.norm(current_force_vec)
        
        # Save data for graphing
        history_steps.append(step)
        history_forces.append(force_mag)
        
        if terminated or truncated:
            print(f"Episode finished at step {step}.")
            break

    server.killServer()

    # 4. Generate the Graph
    print("Generating performance graph...")
    
    # Convert steps to seconds (assuming 0.01s timestep)
    time_seconds = [s * 0.01 for s in history_steps]
    
    plt.figure(figsize=(10, 5))
    
    # Plot the actual force the robot applied
    plt.plot(time_seconds, history_forces, label='Robot Applied Force', color='royalblue', linewidth=2)
    
    # Draw a red dashed line at your target force (10 N)
    plt.axhline(y=10.0, color='red', linestyle='--', label='Target Force (10 N)', linewidth=2)
    
    # Format the graph
    plt.title('HITL Cloned Policy: Force Control Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Contact Force (Newtons)', fontsize=12)
    plt.ylim(0, 25) # Scale Y-axis from 0 to 25 Newtons
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Show the graph!
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_and_graph()