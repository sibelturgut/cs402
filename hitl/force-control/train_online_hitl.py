"""
Online HITL training with hidden human assistance.

The human adds corrections to the executed action, while PPO stores that
executed action as if the policy produced it itself.
"""

import os
import logging
import numpy as np
import torch
import raisimpy as raisim
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from working.config import HitlConfig, DEFAULT_CONFIG
from working.keyboard_handler import KeyboardHandler
from working.panda_env_unified import PandaHitlEnv
from working.joystick_handler import JoystickHandler
from working.online_hitl_wrapper import OnlineHitlWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STATUS_LOG_INTERVAL = 100
CONTROL_LOOP_SLEEP = 0.001

class OnlineHitlTrainer:
    """Train PPO from executed hidden-assist rollouts."""

    def __init__(self, config: HitlConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = None
        self.joystick = None
        self.wrapper = None
        self.model = None
        self.raisim_server = None
        self.total_training_timesteps = 0

        self.episode = 0

    def _build_input_handler(self):
        """Create the configured human input source."""
        device = str(self.config.human_input.input_device).strip().lower()
        if device == "keyboard":
            logger.info("Initializing keyboard control...")
            handler = KeyboardHandler(config=self.config)
            logger.info("✓ Keyboard control ready")
            return handler
        if device == "joystick":
            logger.info("Initializing joystick control...")
            handler = JoystickHandler(config=self.config)
            logger.info("✓ Joystick control ready")
            return handler

        raise ValueError(
            f"Unsupported input_device={self.config.human_input.input_device!r}. "
            "Use 'keyboard' or 'joystick'."
        )

    def _resolve_ppo_batch_size(self) -> int:
        """Choose a batch size that divides the episode rollout cleanly."""
        batch_size = min(self.config.training.ppo_batch_size, self.config.episode_length)
        while batch_size > 2 and self.config.episode_length % batch_size != 0:
            batch_size -= 1
        return max(2, batch_size)

    def _ensure_sb3_training_state(self) -> None:
        """Initialize SB3 internals normally created by `learn()`."""
        if self.model is None or hasattr(self.model, "_logger"):
            return

        total_timesteps = max(1, self.total_training_timesteps or self.config.episode_length)
        self.model._setup_learn(
            total_timesteps=total_timesteps,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="hidden_assist_ppo",
            progress_bar=False,
        )
        logger.info("SB3 manual training state initialized")

    def _build_model(self) -> None:
        """Create the PPO model that is updated once per collected episode."""
        env_sb3 = DummyVecEnv([lambda: self.env])
        batch_size = self._resolve_ppo_batch_size()
        training_cfg = self.config.training

        self.model = PPO(
            "MlpPolicy",
            env_sb3,
            # Keep one rollout buffer aligned with one full episode.
            n_steps=self.config.episode_length,
            batch_size=batch_size,
            n_epochs=training_cfg.ppo_n_epochs,
            learning_rate=training_cfg.ppo_lr,
            gae_lambda=training_cfg.ppo_gae_lambda,
            gamma=training_cfg.ppo_gamma,
            verbose=0,
            device=self.device,
        )
        self._ensure_sb3_training_state()

        logger.info(f"✓ PPO policy initialized on {self.device}")
        logger.info(
            f"  Rollout horizon: {self.config.episode_length} steps per episode | "
            f"Minibatch size: {batch_size}"
        )

    def _log_ready_message(self) -> None:
        """Print the operator instructions shown once after startup."""
        device = str(self.config.human_input.input_device).strip().lower()
        logger.info("")
        logger.info("READY FOR TRAINING!")
        logger.info("=" * 70)
        logger.info("1. Open http://localhost:8080 in your browser")
        if device == "joystick":
            logger.info("2. Use the joystick D-pad or left stick to guide force")
            logger.info("   - UP = move toward 0N (decrease force)")
            logger.info("   - DOWN = move toward -20N (increase negative/push force)")
            if self.config.human_input.enable_haptics:
                logger.info("   - Rumble pulses on target and scales gradually off target")
        else:
            logger.info("2. Focus the pygame control window and use W/S or UP/DOWN")
            logger.info("   - W / UP = move toward 0N (decrease force)")
            logger.info("   - S / DOWN = move toward -20N (increase negative/push force)")
        logger.info("3. Your correction is added invisibly to the policy action")
        logger.info("4. PPO trains from the executed assisted action only")
        logger.info("5. Press Ctrl+C to stop training")
        logger.info("=" * 70)
        logger.info("")

    def _save_model(self, filename: str, message: str) -> None:
        """Save the current PPO weights under the configured model directory."""
        path = os.path.join(self.config.logging.model_dir, filename)
        self.model.save(path)
        logger.info(f"{message}: {path}")

    def _log_step_status(self, step: int, info: dict, reward: float) -> None:
        logger.info(
            f"  Step {step:4d} | "
            f"Human: {info.get('human_accumulated_force', 0.0):+6.2f}N | "
            f"AI Intent: {info.get('ai_intent_force', 0.0):+6.2f}N | "
            f"Total Cmd: {info.get('commanded_force_z', 0.0):+6.2f}N | "
            f"Actual: {info.get('force_mag', 0.0):+6.2f}N | "
            f"Reward: {reward:+.3f}"
        )

    def _log_episode_summary(self, stats: dict) -> None:
        logger.info(
            f"Episode {self.episode} Complete: "
            f"Steps={stats['total_steps']}, "
            f"Reward={stats['total_reward']:.1f}, "
            f"Interventions={stats['interventions']} ({stats['intervention_rate'] * 100:.1f}%), "
            f"Human Force: {stats['human_accumulated_force']:+.2f}N, "
            f"AI Intent: {stats['ai_intent_force']:+.2f}N, "
            f"Total Cmd: {stats.get('total_command_force', stats['total_applied_force']):+.2f}N"
        )

    def _add_rollout_step(
        self,
        rollout_buffer,
        obs: np.ndarray,
        reward: float,
        episode_start: np.ndarray,
        rollout_data: dict,
    ) -> None:
        """Append one environment step to the manual PPO rollout buffer."""
        rollout_buffer.add(
            obs=np.expand_dims(obs, axis=0),
            action=rollout_data["action"],
            reward=np.array([reward], dtype=np.float32),
            episode_start=episode_start,
            value=rollout_data["value"],
            log_prob=rollout_data["log_prob"],
        )
        self.model.num_timesteps += 1

    def _compute_last_values(self, obs: np.ndarray) -> torch.Tensor:
        """Value estimate for the final observation in the rollout."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(self.device)
            return self.model.policy.predict_values(obs_t)

    def _update_policy_from_rollout(
        self,
        rollout_buffer,
        obs: np.ndarray,
        collected_steps: int,
        final_terminated: bool,
        final_truncated: bool,
    ) -> None:
        """Finish GAE/returns for the collected rollout and run one PPO update."""
        if collected_steps != rollout_buffer.buffer_size:
            logger.warning(
                f"Skipping PPO update because rollout is incomplete "
                f"({collected_steps}/{rollout_buffer.buffer_size} steps)"
            )
            return

        dones = np.array([final_terminated], dtype=np.float32)
        if final_truncated and not final_terminated:
            dones = np.array([False], dtype=np.float32)

        rollout_buffer.compute_returns_and_advantage(
            last_values=self._compute_last_values(obs),
            dones=dones,
        )
        self.model._update_current_progress_remaining(
            self.model.num_timesteps,
            max(1, self.total_training_timesteps),
        )
        self.model.train()
        logger.info("PPO update complete from executed assisted rollout")
    
    def initialize(self) -> None:
        """Initialize environment, controls, policy, and server."""
        logger.info("=" * 70)
        logger.info("ONLINE HITL TRAINING - HIDDEN HUMAN ASSISTANCE")
        logger.info("=" * 70)
        
        try:
            logger.info("Initializing Panda environment...")
            self.env = PandaHitlEnv(config=self.config)
            logger.info("✓ Environment ready")

            self.joystick = self._build_input_handler()

            logger.info("Initializing PPO policy...")
            self._build_model()

            logger.info("Initializing HITL wrapper...")
            self.wrapper = OnlineHitlWrapper(
                env=self.env,
                joystick_handler=self.joystick,
                policy=self.model.policy,
                config=self.config,
            )
            logger.info("✓ HITL wrapper ready")
            
            logger.info(f"Starting Raisim server on port {self.config.raisim_server_port}...")
            self.raisim_server = raisim.RaisimServer(self.env.world)
            self.raisim_server.launchServer(self.config.raisim_server_port)
            logger.info(f"✓ Raisim server at http://localhost:{self.config.raisim_server_port}")
            
            self._log_ready_message()
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.cleanup()
            raise
    
    def train_episode(self) -> dict:
        """Run one assisted episode and update PPO from that rollout."""
        self.episode += 1
        self._ensure_sb3_training_state()
        
        obs = self.wrapper.reset_episode()
        rollout_buffer = self.model.rollout_buffer
        rollout_buffer.reset()
        # SB3 expects an "episode starts here" flag for the first stored step.
        episode_start = np.ones((1,), dtype=np.float32)
        collected_steps = 0
        final_terminated = False
        final_truncated = False
        
        logger.info(f"\n--- Episode {self.episode} Started ---")
        
        for step in range(self.config.episode_length):
            try:
                obs_before = obs.copy()
                obs, reward, terminated, truncated, info, rollout_data = self.wrapper.step_hitl(obs)
                self._add_rollout_step(
                    rollout_buffer=rollout_buffer,
                    obs=obs_before,
                    reward=reward,
                    episode_start=episode_start,
                    rollout_data=rollout_data,
                )
                collected_steps += 1
                
                if (step + 1) % STATUS_LOG_INTERVAL == 0:
                    self._log_step_status(step + 1, info, reward)
                
                # The next transition starts a new episode only if this step ended one.
                episode_start = np.array([terminated or truncated], dtype=np.float32)
                final_terminated = terminated
                final_truncated = truncated
                if terminated or truncated:
                    break
                
                time.sleep(CONTROL_LOOP_SLEEP)
                
            except Exception as e:
                logger.error(f"Error at step {step}: {e}", exc_info=True)
                break
        
        stats = self.wrapper.get_episode_stats()
        self._log_episode_summary(stats)
        self._update_policy_from_rollout(
            rollout_buffer=rollout_buffer,
            obs=obs,
            collected_steps=collected_steps,
            final_terminated=final_terminated,
            final_truncated=final_truncated,
        )
        
        return stats
    
    def run(self, num_episodes: int | None = None):
        """Run the configured number of training episodes."""
        if num_episodes is None:
            num_episodes = self.config.num_episodes

        try:
            self.total_training_timesteps = max(1, num_episodes * self.config.episode_length)
            self.initialize()
            logger.info(
                f"Training horizon: {num_episodes} episodes / "
                f"{self.total_training_timesteps} total timesteps"
            )
            
            for ep in range(num_episodes):
                self.train_episode()
                
                if (ep + 1) % self.config.logging.checkpoint_interval == 0:
                    self._save_model(
                        filename=f"online_hitl_ep{ep+1}.zip",
                        message="Checkpoint saved",
                    )
            
            self._save_model("online_hitl_final.zip", "Final model saved")
            logger.info("Training complete!")
            
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")

        resources = (
            (self.raisim_server, "killServer", "✓ Raisim server stopped"),
            (self.joystick, "close", "✓ Joystick closed"),
            (self.env, "close", "✓ Environment closed"),
        )
        for resource, method_name, message in resources:
            if resource is None:
                continue
            try:
                getattr(resource, method_name)()
                logger.info(message)
            except Exception:
                pass
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    config = DEFAULT_CONFIG
    
    # Optional example overrides:
    # config.training.ppo_lr = 1e-4
    # config.episode_length = 500
    # config.num_episodes = 20
    
    trainer = OnlineHitlTrainer(config=config)
    trainer.run()


if __name__ == "__main__":
    main()
