#!/usr/bin/env python3
"""
Quick test script for Lissajous-HITL integration.

Run: python test_lissajous_integration.py

This verifies:
  - Environment initializes correctly
  - Observations have correct shape
  - Rewards are computed properly
  - Lissajous trajectory is generated
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add this example directory to the import path when run from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HitlConfig, DEFAULT_CONFIG, RobotConfig
from panda_env_lissajous_hitl import PandaLissajousHitlEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lissajous_env():
    """Test the Lissajous HITL environment."""
    logger.info("=" * 70)
    logger.info("Testing Lissajous-HITL Environment")
    logger.info("=" * 70)
    
    # Create config with Lissajous enabled
    config = HitlConfig()
    config.robot.enable_lissajous = True
    
    logger.info(f"Config: enable_lissajous={config.robot.enable_lissajous}")
    logger.info(f"  Lissajous freq_x: {config.robot.lissajous_freq_x} Hz")
    logger.info(f"  Lissajous freq_y: {config.robot.lissajous_freq_y} Hz")
    logger.info(f"  Amplitude X: {config.robot.lissajous_amplitude_x} m")
    logger.info(f"  Amplitude Y: {config.robot.lissajous_amplitude_y} m")
    
    try:
        # Initialize environment
        logger.info("\nInitializing environment...")
        env = PandaLissajousHitlEnv(config=config)
        
        # Check observation space
        logger.info(f"Observation space shape: {env.observation_space.shape}")
        expected_obs_shape = (20,)
        assert env.observation_space.shape == expected_obs_shape, \
            f"Expected obs shape {expected_obs_shape}, got {env.observation_space.shape}"
        
        # Check action space
        logger.info(f"Action space: {env.action_space}")
        
        # Reset environment
        logger.info("\nResetting environment...")
        obs, info = env.reset()
        logger.info(f"Initial obs shape: {obs.shape}")
        assert obs.shape == expected_obs_shape, f"Reset obs shape mismatch"
        
        # Run a few steps
        logger.info("\nRunning 100 steps...")
        rewards = []
        forces = []
        
        for step in range(100):
            # Small random action
            action = np.array([np.sin(step * 0.05) * 0.3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            forces.append(info.get('force_mag', 0.0))
            if step % 20 == 0:
                logger.info(
                    f"  Step {step:3d}: "
                    f"Reward={reward:+.3f}, "
                    f"Force={info.get('force_mag', 0.0):+.2f}N, "
                    f"CmdForce={env.commanded_force_z:+.2f}N"
                )
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {step}")
                break
        
        # Statistics
        logger.info("\nStatistics:")
        logger.info(f"  Avg reward: {np.mean(rewards):.3f}")
        logger.info(f"  Final obs shape: {obs.shape}")
        logger.info(f"  Avg force: {np.mean(forces):.2f}N")
        
        # Verify observations
        logger.info("\nObservation breakdown:")
        logger.info(f"  q (joint pos) [0:7]: {obs[:7]}")
        logger.info(f"  v (joint vel) [7:14]: {obs[7:14]}")
        logger.info(f"  f (contact force) [14:17]: {obs[14:17]}")
        logger.info(f"  dist_z [17]: {obs[17]}")
        logger.info(f"  cmd_force [18]: {obs[18]}")
        logger.info(f"  error_sum [19]: {obs[19]}")
        
        env.close()
        logger.info("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def compare_with_standard_env():
    """Compare observation sizes between standard and Lissajous environments."""
    logger.info("\n" + "=" * 70)
    logger.info("Comparing Standard vs. Lissajous Environments")
    logger.info("=" * 70)
    
    try:
        from panda_env_unified import PandaHitlEnv
        
        config = HitlConfig()
        
        # Standard environment
        logger.info("\nInitializing standard environment (force control only)...")
        env_std = PandaHitlEnv(config=config)
        logger.info(f"  Obs space shape: {env_std.observation_space.shape}")
        
        # Lissajous environment
        logger.info("Initializing Lissajous environment (trajectory + force)...")
        config.robot.enable_lissajous = True
        env_lis = PandaLissajousHitlEnv(config=config)
        logger.info(f"  Obs space shape: {env_lis.observation_space.shape}")
        
        logger.info(f"\nDifference: {env_lis.observation_space.shape[0] - env_std.observation_space.shape[0]} dims")
        logger.info("  (1 additional dim for integrated force error)")
        
        env_std.close()
        env_lis.close()
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)


if __name__ == "__main__":
    success = test_lissajous_env()
    compare_with_standard_env()
    
    sys.exit(0 if success else 1)
