import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import raisimpy as raisim
import os

class PandaEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.control_dt = 0.01  # Your AI still runs at 100Hz
        self.physics_dt = 0.002 # Physics runs 5x faster (500Hz)
        self.sim_substeps = int(self.control_dt / self.physics_dt)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float64)
        
        # --- RAISIM SETUP ---
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../../rsc/activation.raisim")
        self.world = raisim.World()
        self.world.addGround()
        self.world.setTimeStep(self.physics_dt)
        
        self.panda = self.world.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../../../rsc/panda/urdf/panda.urdf")
        self.panda.setName("panda")

        self.box_pos = [0.2, -0.45, 0.1]
        self.box = self.world.addBox(0.4, 0.4, 0.5, 100.0)
        self.box.setPosition(*self.box_pos)
        self.box.setBodyType(raisim.BodyType.STATIC)

        self.target_force_mag = 10.0 
        self.ee_frame_idx = self.panda.getFrameIdxByName("EndEffector_Sphere")
        self.ee_frame_name = "EndEffector_Sphere"
        self.ee_offset = 0.014142 # The radius offset you mentioned

        # Stabilized contact configuration
        self.contact_q = np.array([-0.998263, 0.05, -0.0535692, -2.0964, 0.0846695, 2.45698, 0.67558])
        # We need a variable to "remember" the previous force command
        self.commanded_force_z = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the command to something safe (e.g., 0 or slight push)
        self.commanded_force_z = 0.0 
        
        self.panda.setGeneralizedCoordinate(self.contact_q)
        self.panda.setGeneralizedVelocity(np.zeros(7))
        self.panda.setGeneralizedForce(np.zeros(7))
        return self._get_obs(), {}

    def step(self, action):
        action_val = float(action[0])
        if abs(action_val) < 0.01: action_val = 0.0
            
        # 1. Update the command
        # Moving in air is slow, pushing on box is normal
        current_force = self._get_contact_force()
        force_mag = np.linalg.norm(current_force)
        
        speed_mult = 0.15 if force_mag < 0.1 else 1.0
        self.commanded_force_z += (action_val * 0.4 * speed_mult)
        self.commanded_force_z = np.clip(self.commanded_force_z, -20.0, 10.0)
        
        # 2. Prepare Force Command
        desired_force_at_ee = np.array([0.0, 0.0, self.commanded_force_z]) 
        
        # --- PHYSICS INTEGRATION LOOP ---
        for _ in range(self.sim_substeps):
            # A. Get Jacobian for the Force Control
            J = self.panda.getDenseFrameJacobian(self.ee_frame_name)
            tau_action = np.matmul(J[:3, :].T, desired_force_at_ee)

            # B. Joint Impedance (The "Springs")
            # This keeps the arm from "flopping" in the air
            q = self.panda.getGeneralizedCoordinate()
            v = self.panda.getGeneralizedVelocity()
            
            # If in air, use strong springs. If on box, use very weak springs.
            kp_val = 40.0 if force_mag < 0.1 else 5.0 
            kd_val = 10.0 if force_mag < 0.1 else 2.0
            
            # Pull joints toward the 'contact_q' pose
            tau_spring = kp_val * (self.contact_q - q) - kd_val * v
            
            # C. Gravity Comp
            tau_grav = self.panda.getNonlinearities(self.world.getGravity())
            
            # Combine everything
            # tau_action handles the Z-force, tau_spring handles the "weird" flopping
            self.panda.setGeneralizedForce(tau_action + tau_grav + tau_spring)
            self.world.integrate()
        
        # Calculate Reward & Return (Keep your existing reward logic)
        force_error = abs(force_mag - self.target_force_mag)
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist_to_surface = max(0, ee_pos[2] - 0.35 - self.ee_offset)
        reward = np.exp(-1.0 * force_error) if dist_to_surface < 0.002 else -50.0 * dist_to_surface

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        q = self.panda.getGeneralizedCoordinate()
        v = self.panda.getGeneralizedVelocity()
        f = self._get_contact_force()
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist = max(0, (ee_pos[2] - 0.35) - self.ee_offset)
        return np.concatenate([q, v, f, [dist], [self.commanded_force_z]])

    def _get_contact_force(self):
        contacts = self.panda.getContacts()
        total_force = np.zeros(3)
        for c in contacts:
            if not c.skip():
                total_force += np.array(c.getImpulse())
        return total_force / self.physics_dt