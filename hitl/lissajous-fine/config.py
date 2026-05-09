"""Configuration for the active hidden-assistance HITL training setup."""

import os
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    """Simulation timing shared by the environment and control loop."""

    control_dt: float = 0.01
    physics_dt: float = 0.001
    # Number of physics integration steps executed for each control action.
    substeps: int = 10
    gravity: tuple = (0.0, 0.0, -9.81)
    
@dataclass
class RobotConfig:
    """Task-level robot settings for vertical force control and trajectory tracking."""

    target_force_mag: float = 10.0
    # Commanded end-effector z-force range. `0` releases contact, `-20` pushes hardest.
    max_commanded_force: float = 0.0
    min_commanded_force: float = -20.0
    ee_frame_name: str = "EndEffector_Sphere"
    ee_offset: float = 0.014142
    surface_height: float = 0.35
    
    # Impedance gains used by the low-level stabilizing controller in simulation.
    kp_spring_in_air: float = 40.0
    kd_damper_in_air: float = 10.0
    kp_spring_on_surface: float = 5.0
    kd_damper_on_surface: float = 2.0
    force_threshold_in_air: float = 0.1
    
    # ========== Lissajous trajectory tracking (optional) ==========
    # Enable Lissajous trajectory following in addition to force control
    enable_lissajous: bool = False
    
    # Lissajous curve parameters (Figure-8 pattern with 1:2 frequency ratio)
    lissajous_freq_x: float = 0.2   # Hz (5 second period)
    lissajous_freq_y: float = 0.4   # Hz (2.5 second period, 2:1 ratio)
    lissajous_amplitude_x: float = 0.03  # meters
    lissajous_amplitude_y: float = 0.03  # meters
    box_center_x: float = 0.2       # Center of trajectory X
    box_center_y: float = -0.45     # Center of trajectory Y
    
    # Spring-damper gains for X-Y trajectory tracking
    kp_trajectory_xy: float = 800.0
    kd_trajectory_xy: float = 30.0

@dataclass
class HumanInputConfig:
    """Operator input and haptic-feedback settings."""

    # Select `"keyboard"` or `"joystick"` once here instead of editing the trainer.
    input_device: str = "keyboard"
    sensitivity: float = 1.0
    # Values below this are treated as "no human correction".
    deadzone: float = 0.1
    # Executed action = clip(policy_action + assist_gain * human_action, -1, 1)
    assist_gain: float = 0.5

    # Joystick-only haptic feedback. Keyboard mode ignores these fields.
    enable_haptics: bool = True
    # Entering this error band triggers a short confirmation pulse.
    on_target_band: float = 0.5
    on_target_pulse_low: float = 0.20
    on_target_pulse_high: float = 0.35
    on_target_pulse_ms: int = 120
    # Continuous off-target rumble starts just outside this band.
    off_target_min_error: float = 0.5
    # Approximate error where the Gaussian falloff has mostly faded to the floor.
    off_target_max_error: float = 8.0
    # Low background rumble when the robot is far from target.
    off_target_min_rumble: float = 0.05
    # Peak off-target rumble just outside the on-target band.
    off_target_max_rumble: float = 0.20
    # Avoid resending a rumble command every control tick.
    rumble_refresh_ms: int = 150

@dataclass
class TrainingConfig:
    """PPO hyperparameters used by the current hidden-assistance loop."""

    ppo_lr: float = 3e-4
    # Minibatch size inside one PPO update. The trainer reduces it if needed so
    # it divides the episode rollout cleanly.
    ppo_batch_size: int = 64
    # Number of optimizer passes over each collected episode rollout.
    ppo_n_epochs: int = 15
    ppo_gae_lambda: float = 0.95
    ppo_gamma: float = 0.99

@dataclass
class LoggingConfig:
    """Output locations and checkpoint cadence."""

    log_dir: str = "logs/hitl"
    model_dir: str = "logs/hitl/models"
    data_dir: str = "logs/hitl/data"
    checkpoint_interval: int = 5
    
    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

@dataclass
class HitlConfig:
    """Top-level configuration for the active training script."""

    physics: PhysicsConfig = None
    robot: RobotConfig = None
    human_input: HumanInputConfig = None
    training: TrainingConfig = None
    logging: LoggingConfig = None
    
    # One PPO rollout is collected per episode.
    episode_length: int = 1000
    num_episodes: int = 30
    raisim_server_port: int = 8080

    def __post_init__(self):
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.robot is None:
            self.robot = RobotConfig()
        if self.human_input is None:
            self.human_input = HumanInputConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

DEFAULT_CONFIG = HitlConfig()
