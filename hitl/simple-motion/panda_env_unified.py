"""Single-task Panda force-control environment used by the HITL trainer."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import raisimpy as raisim
import os
import logging
from config import HitlConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class PandaHitlEnv(gym.Env):
    """
    Panda force-control environment with a fixed contact-start configuration.
    
    Observations (19-dim):
        - Joint positions (7)
        - Joint velocities (7)
        - Measured contact force (3)
        - Distance to surface (1)
        - Current commanded z-force (1)
    
    Action:
        One scalar in [-1, 1] that changes the commanded z-force by a small
        amount each control step. Zero means "hold current commanded force".
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config: HitlConfig = None):
        """Initialize the simulation world and task state."""
        self.config = config or DEFAULT_CONFIG
        self.physics_cfg = self.config.physics
        self.robot_cfg = self.config.robot
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        
        self._init_raisim()
        
        # Force-control state used by the trainer and UI.
        self.commanded_force_z = 0.0
        self.human_accumulated_force = 0.0
        self.steps_count = 0
        # Fixed joint configuration that starts the end-effector in contact.
        # Centred over the box's top surface (box top-centre is X=0.2, Y=-0.45)
        # with a small clearance so the robot settles cleanly instead of
        # spawning embedded in the surface (which causes violent bounce-offs).
        self.contact_q = np.array([
            -1.04003354, -0.25058416, -0.13173775, -2.39598415,
            0.13307223, 2.33590453, 0.69235052
        ])
        # World X/Y position of the EE at the start of each episode; the Lissajous
        # figure-8 oscillates around this point on the surface plane.
        self.ee_x0 = 0.0
        self.ee_y0 = 0.0
        
        logger.info(f"PandaHitlEnv initialized with {self.physics_cfg.substeps} substeps")
    
    def _init_raisim(self):
        """Create the Raisim world, Panda robot, and contact surface."""
        license_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../../rsc/activation.raisim"
        urdf_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../../rsc/panda/urdf/panda.urdf"
        
        if not os.path.exists(license_path):
            logger.warning(f"License file not found: {license_path}")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        raisim.World.setLicenseFile(license_path)
        
        self.world = raisim.World()
        self.world.addGround()
        self.world.setTimeStep(self.physics_cfg.physics_dt)
        
        self.panda = self.world.addArticulatedSystem(urdf_path)
        self.panda.setName("panda")
        
        self.box = self.world.addBox(0.4, 0.4, 0.5, 100.0)
        self.box.setPosition(0.2, -0.45, 0.1)
        self.box.setBodyType(raisim.BodyType.STATIC)
        
        self.ee_frame_idx = self.panda.getFrameIdxByName(self.robot_cfg.ee_frame_name)
        logger.info(f"Raisim initialized. EE frame idx: {self.ee_frame_idx}")
    
    def reset(self, seed=None, options=None):
        """Reset the robot to the fixed contact-start state."""
        super().reset(seed=seed)
        
        self.commanded_force_z = self.robot_cfg.initial_push_force
        self.human_accumulated_force = 0.0
        self.steps_count = 0

        self.panda.setGeneralizedCoordinate(self.contact_q)
        self.panda.setGeneralizedVelocity(np.zeros(7))
        self.panda.setGeneralizedForce(np.zeros(7))

        # Set ee_x0/y0 before settling so the sinusoidal XY target = current EE
        # position during settle (zero XY error, no unwanted lateral force).
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        self.ee_x0 = float(ee_pos[0])
        self.ee_y0 = float(ee_pos[1])

        # Settle with the initial push force actually applied so the arm reaches
        # stable contact before PPO takes control (raw integrate() applies no force).
        for _ in range(50):
            f_mag = np.linalg.norm(self._get_contact_force())
            self._physics_step(f_mag)

        # Re-capture after settling in case the arm drifted slightly.
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        self.ee_x0 = float(ee_pos[0])
        self.ee_y0 = float(ee_pos[1])

        obs = self._get_obs()
        return obs.astype(np.float32), {}
    
    def step(self, action):
        """Apply one force-delta action and step the simulation."""
        action_val = float(action[0])
        
        # Delta control: zero action preserves the previous commanded force.
        delta_force = action_val * 20.0 * 0.01
        self.commanded_force_z += delta_force
        
        self.commanded_force_z = np.clip(
            self.commanded_force_z,
            self.robot_cfg.min_commanded_force,
            self.robot_cfg.max_commanded_force
        )
        
        for _ in range(self.physics_cfg.substeps):
            current_force = self._get_contact_force()
            force_mag = np.linalg.norm(current_force)
            self._physics_step(force_mag)
        
        self.steps_count += 1
        obs = self._get_obs()
        
        # Dense reward on surface, strong penalty when the robot lifts off.
        force_mag = np.linalg.norm(self._get_contact_force())
        force_error = abs(force_mag - self.robot_cfg.target_force_mag)
        
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist_to_surface = max(
            0, 
            ee_pos[2] - self.robot_cfg.surface_height - self.robot_cfg.ee_offset
        )
        
        if dist_to_surface < 0.002:
            reward_raw = 1.0 - (force_error / max(1e-6, self.robot_cfg.target_force_mag))
            reward = float(np.clip(reward_raw, -1.0, 1.0))
        else:
            reward = -100.0 * dist_to_surface
        
        truncated = self.steps_count >= self.config.episode_length
        terminated = False
        
        info = {
            'force_mag': float(force_mag),
            'force_error': float(force_error),
            'commanded_force_z': float(self.commanded_force_z),
            'total_command_force': float(self.commanded_force_z),
            'human_accumulated_force': float(self.human_accumulated_force),
            'total_applied_force': float(force_mag),
            'on_surface': bool(dist_to_surface < 0.002),
            'reward': float(reward),
        }
        
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def _physics_step(self, force_mag: float):
        """Advance one physics step with force command + impedance stabilization."""
        J = self.panda.getDenseFrameJacobian(self.robot_cfg.ee_frame_name)
        J_lin = J[:3, :]  # 3×n_dof — maps joint velocities to EE linear velocity

        q = self.panda.getGeneralizedCoordinate()
        v = self.panda.getGeneralizedVelocity()

        # --- Gravity compensation ---
        tau_grav = self.panda.getNonlinearities(self.world.getGravity())

        # --- Z-force control (PPO action drives this) ---
        tau_z = np.matmul(J_lin.T, np.array([0.0, 0.0, self.commanded_force_z]))

        # --- Lissajous figure-8 motion on the surface plane (world X-Y) ---
        # x(t) = Ax*sin(wt), y(t) = Ay*sin(2wt) -> classic "infinity" curve.
        # NOTE: use the integrated simulation clock (advances every substep),
        # not steps_count*control_dt -- the latter is frozen across all
        # `substeps` physics integrations within one control step, producing a
        # staircased reference (position holds, then jumps every 10ms) that the
        # Cartesian PD fights and which leaks into the Z-contact force as jitter.
        t = self.world.getWorldTime()
        omega = 2 * np.pi * self.robot_cfg.lissajous_frequency
        Ax = self.robot_cfg.lissajous_amplitude_x

        x_des = self.ee_x0 + Ax * np.sin(omega * t)
        y_des = self.ee_y0
        vx_des = Ax * omega * np.cos(omega * t)
        vy_des = 0.0

        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        ee_vel_xy = np.matmul(J_lin[:2, :], v)  # actual EE velocity [vx, vy]

        F_x = (
            self.robot_cfg.lissajous_kp * (x_des - float(ee_pos[0]))
            + self.robot_cfg.lissajous_kd * (vx_des - float(ee_vel_xy[0]))
        )
        F_y = (
            self.robot_cfg.lissajous_kp * (y_des - float(ee_pos[1]))
            + self.robot_cfg.lissajous_kd * (vy_des - float(ee_vel_xy[1]))
        )

        # Coulomb friction feedforward: the surface exerts up to μ·|F_z| friction
        # opposing motion.  Compensate along the direction of desired travel.
        speed_des = float(np.hypot(vx_des, vy_des))
        if speed_des > 1e-6:
            f_comp = self.robot_cfg.lissajous_friction_comp * abs(self.commanded_force_z)
            F_x += f_comp * (vx_des / speed_des)
            F_y += f_comp * (vy_des / speed_des)

        tau_motion = J_lin[0, :] * F_x + J_lin[1, :] * F_y  # world X/Y force -> joint torques

        # Soft joint spring: blend smoothly between in-air and on-surface gains
        # over 0–10 N contact force. A hard binary switch at 0.1 N (old code)
        # caused 8× stiffness jumps at 1 kHz inside the substep loop, creating
        # limit cycles every time lateral motion briefly unloaded the contact.
        blend = float(np.clip(force_mag / 10.0, 0.0, 1.0))
        kp = self.robot_cfg.kp_spring_in_air * (1.0 - blend) + self.robot_cfg.kp_spring_on_surface * blend
        kd = self.robot_cfg.kd_damper_in_air * (1.0 - blend) + self.robot_cfg.kd_damper_on_surface * blend
        tau_spring = kp * (self.contact_q - q) - kd * v

        self.panda.setGeneralizedForce(tau_grav + tau_z + tau_motion + tau_spring)
        self.world.integrate()
    
    def _get_obs(self) -> np.ndarray:
        """Assemble the 19D observation used by PPO."""
        q = self.panda.getGeneralizedCoordinate()
        v = self.panda.getGeneralizedVelocity()
        f = self._get_contact_force()
        
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist = max(
            0,
            (ee_pos[2] - self.robot_cfg.surface_height) - self.robot_cfg.ee_offset
        )
        
        obs = np.concatenate([
            q,                         # 7 dims
            v,                         # 7 dims
            f,                         # 3 dims
            [dist],                    # 1 dim
            [self.commanded_force_z]   # 1 dim
        ])
        return obs
    
    def _get_contact_force(self) -> np.ndarray:
        """Sum all active contact impulses and convert them to force."""
        contacts = self.panda.getContacts()
        total_force = np.zeros(3)
        for c in contacts:
            if not c.skip():
                total_force += np.array(c.getImpulse())
        return total_force / self.physics_cfg.physics_dt
    
    def close(self):
        """Release the Raisim world."""
        if hasattr(self, 'world'):
            del self.world
        logger.info("Environment closed")


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    env = PandaHitlEnv()
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.4f}, Force cmd: {env.commanded_force_z:.2f}")
    
    env.close()
