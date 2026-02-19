import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import raisimpy as raisim
from panda_env_force_delta import PandaEnv
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

TOTAL_TIMESTEPS = 600000

def run():
    env = PandaEnv()
    env = TimeLimit(env, max_episode_steps=1000)

    eval_env = PandaEnv()
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)
    eval_env = Monitor(eval_env)

    log_dir = f"logs/force/PPO_model/"
    model_path = os.path.join(log_dir, "model.zip")
    os.makedirs(log_dir, exist_ok=True)

    server = raisim.RaisimServer(env.unwrapped.world)
    server.launchServer(8080)

    print("Server launched. Press Enter to start...")
    input()

    if os.path.exists(model_path):
        print(f"--- Loading existing model ---")
        model = PPO.load(model_path, env=env)
    else:
        print(f"--- Training new model ---")
        env = Monitor(env, log_dir)
        
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/force/checkpoints/')
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/force/best_model/', eval_freq=5000)
        
        model = PPO(
            "MlpPolicy",
            env,  
            n_steps=2048,
            batch_size=128,
            ent_coef=0.001,      # Slight exploration
            gamma=0.99,
            learning_rate=1e-4,  # Lowered for precision
            verbose=1,
            tensorboard_log="./logs/force/tensorboard/"
        )
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([checkpoint_callback, eval_callback]))
        model.save(model_path)

    # --- MANUAL EVALUATION ---
    print("--- Starting Evaluation ---")
    history_steps = []
    history_force_mag = []
    
    obs, _ = env.reset()
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Force is at -4:-1, Distance is at -1
        current_force = obs[-4:-1]
        force_mag = np.linalg.norm(current_force)
        
        history_steps.append(step)
        history_force_mag.append(force_mag)

        if step % 50 == 0:
            print(f"Step {step}: Force {force_mag:.2f} N | Dist {obs[-1]:.4f}m")
        
        if terminated or truncated: break

    plt.plot(history_steps, history_force_mag, label="Force Magnitude")
    plt.axhline(y=10.0, color='r', linestyle='--', label="Target 10N")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.savefig("force_result.png")
    plt.show()

    server.killServer()

if __name__ == '__main__':
    run()