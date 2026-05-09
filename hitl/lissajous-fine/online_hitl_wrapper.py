"""
Hidden-assistance HITL wrapper.

Human corrections are applied additively to the policy action before execution.
The executed action is what PPO later learns from; there is no behavioral cloning
or explicit human-labeled supervision path.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class OnlineHitlWrapper:
    """Sample actions, apply hidden human corrections, and return PPO rollout data."""

    def __init__(
        self,
        env,
        joystick_handler,
        policy,
        config=None,
    ):
        self.env = env
        self.joystick = joystick_handler
        self.policy = policy
        self.config = config

        # Episode stats
        self.episode = 0
        self.step_count = 0
        self.interventions = 0
        self.episode_reward = 0.0
        self.last_force = 0.0

    def reset_episode(self) -> np.ndarray:
        """Reset episode bookkeeping and environment state."""
        self.episode += 1
        self.step_count = 0
        self.interventions = 0
        self.episode_reward = 0.0
        self.last_force = 0.0

        if hasattr(self.env, "human_accumulated_force"):
            self.env.human_accumulated_force = 0.0

        obs, _ = self.env.reset()
        return obs

    def _sample_policy_action(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Sample the current policy and return the action used as the AI intent."""
        self.policy.set_training_mode(False)
        device = getattr(self.policy, "device", torch.device("cpu"))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(device)
            sampled_action, values, _ = self.policy(obs_t)

        ai_action = float(np.clip(sampled_action.cpu().numpy().ravel()[0], -1.0, 1.0))
        return obs_t, values, ai_action

    def _get_human_action(self, force_mag: float) -> float:
        """Read the latest human correction from the keyboard handler."""
        return float(self.joystick.get_action(force_mag)[0])

    def _get_executed_action(self, ai_action: float, human_action: float) -> float:
        """Apply the hidden additive correction to the sampled policy action."""
        if self.config and hasattr(self.config, 'human_input'):
            assist_gain = float(getattr(self.config.human_input, "assist_gain", 0.5))
        else:
            assist_gain = 0.5
        assist_delta = float(np.clip(human_action * assist_gain, -1.0, 1.0))
        return float(np.clip(ai_action + assist_delta, -1.0, 1.0))

    def _evaluate_action_log_prob(self, obs_t: torch.Tensor, action: float) -> torch.Tensor:
        """Evaluate the log-prob of the executed action, not the raw policy sample."""
        action_t = torch.as_tensor([[action]], dtype=torch.float32, device=obs_t.device)
        with torch.no_grad():
            _, log_prob, _ = self.policy.evaluate_actions(obs_t, action_t)
        return log_prob

    def _get_force_state(self) -> Tuple[float, float, float]:
        """Read force bookkeeping values shared by logs and the UI."""
        human_force = float(getattr(self.env, "human_accumulated_force", 0.0))
        commanded_force = float(getattr(self.env, "commanded_force_z", 0.0))
        ai_intent_force = commanded_force - human_force
        return human_force, commanded_force, ai_intent_force

    def _update_ui(self, human_force: float, ai_intent_force: float) -> None:
        """Refresh optional keyboard/pygame feedback."""
        if self.joystick is None:
            return

        self.joystick.human_accumulated_force = human_force
        self.joystick.ai_intent_force = ai_intent_force
        self.joystick.autonomy_enabled = True

        try:
            self.joystick.update_display(self.last_force)
        except Exception:
            pass

    def step_hitl(self, obs: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict, Dict]:
        """Sample policy, add hidden human correction, and step the environment."""
        obs_t, values, ai_action = self._sample_policy_action(obs)
        # Both HITL envs store measured contact force after q(7) and v(7).
        force_mag = float(np.linalg.norm(obs[14:17]))
        human_action = self._get_human_action(force_mag)
        executed_action = self._get_executed_action(ai_action, human_action)
        log_prob = self._evaluate_action_log_prob(obs_t, executed_action)

        # Get deadzone threshold
        if self.config and hasattr(self.config, 'human_input'):
            deadzone = self.config.human_input.deadzone
        else:
            deadzone = 0.1
        
        is_intervening = abs(human_action) > deadzone
        if is_intervening:
            self.interventions += 1

        assist_action = executed_action - ai_action
        executed_action_array = np.array([executed_action], dtype=np.float32)

        try:
            # Bookkeeping only: the environment dynamics use `executed_action_array`.
            self.env.update_human_force_delta(assist_action)
        except Exception:
            pass

        next_obs, reward, terminated, truncated, info = self.env.step(executed_action_array)

        self.episode_reward += float(reward)
        self.last_force = float(info.get("force_mag", 0.0))
        self.step_count += 1

        human_force, commanded_force, ai_intent_force = self._get_force_state()
        if self.config and hasattr(self.config, "robot") and hasattr(self.config, "physics"):
            force_scale = abs(self.config.robot.min_commanded_force) * self.config.physics.control_dt
        else:
            force_scale = 20.0 * 0.01
        assist_force_delta = -assist_action * force_scale

        info.update(
            {
                "human_intervening": is_intervening,
                "human_action": human_action,
                "assist_delta_action": assist_action,
                "assist_delta_force": assist_force_delta,
                "ai_action": ai_action,
                "executed_action": executed_action,
                "combined_action": executed_action,
                "human_accumulated_force": human_force,
                "ai_intent_force": ai_intent_force,
                "commanded_force_z": commanded_force,
                "hidden_assistance": True,
            }
        )

        self._update_ui(human_force, ai_intent_force)

        rollout_data = {
            "action": executed_action_array,
            "value": values,
            "log_prob": log_prob,
        }

        return next_obs, reward, terminated, truncated, info, rollout_data

    def get_episode_stats(self) -> Dict:
        """Return episode-level metrics for logging."""
        intervention_rate = self.interventions / max(1, self.step_count)
        human_acc = float(getattr(self.env, "human_accumulated_force", 0.0))
        commanded = float(getattr(self.env, "commanded_force_z", 0.0))

        return {
            "episode": self.episode,
            "total_steps": self.step_count,
            "interventions": self.interventions,
            "intervention_rate": intervention_rate,
            "total_reward": float(self.episode_reward),
            "avg_reward": float(self.episode_reward / max(1, self.step_count)),
            "human_accumulated_force": human_acc,
            "ai_intent_force": commanded - human_acc,
            "total_command_force": commanded,
            # Kept as a legacy alias for older log code.
            "total_applied_force": commanded,
        }
