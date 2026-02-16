import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import raisimpy as raisim
import os

class PandaEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.timestep = 0.01 
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float64)
        
        # --- RAISIM SETUP ---
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
        self.world = raisim.World()
        self.world.addGround()
        self.world.setTimeStep(self.timestep)
        
        self.panda = self.world.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/panda/urdf/panda.urdf")
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.panda.setGeneralizedCoordinate(self.contact_q)
        self.panda.setGeneralizedVelocity(np.zeros(7))
        self.panda.setGeneralizedForce(np.zeros(7))
        return self._get_obs(), {}

    def step(self, action):
        # Action mapping: agent [-1, 1] -> force [-30, 10]. 
        # Focusing the range closer to the 10N target helps stability.
        raw_action = float(action[0])
        force_z = (raw_action * 20.0) - 10.0 
        
        desired_force_at_ee = np.array([0.0, 0.0, force_z]) 
        J = self.panda.getDenseFrameJacobian(self.ee_frame_name)
        tau_action = np.matmul(J[:3, :].T, desired_force_at_ee)
       
        tau_grav = self.panda.getNonlinearities(self.world.getGravity())
        tau_damping = -1.0 * self.panda.getGeneralizedVelocity() 

        self.panda.setGeneralizedForce(tau_action + tau_grav + tau_damping)
        self.world.integrate()
        
        # --- REWARD ---
        current_force = self._get_contact_force()
        force_mag = np.linalg.norm(current_force)
        force_error = abs(force_mag - self.target_force_mag)
        
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        # Corrected distance with offset
        dist_to_surface = max(0, (ee_pos[2] - 0.35) - self.ee_offset)

        if dist_to_surface > 0.002:
            # Phase 1: Moving down (always negative)
            reward = -100.0 * dist_to_surface
        else:
            reward = np.exp(-1.0 * force_error)

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        q = self.panda.getGeneralizedCoordinate()
        v = self.panda.getGeneralizedVelocity()
        f = self._get_contact_force()
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist = max(0, (ee_pos[2] - 0.35) - self.ee_offset)
        return np.concatenate([q, v, f, [dist]])

    def _get_contact_force(self):
        contacts = self.panda.getContacts()
        total_force = np.zeros(3)
        for c in contacts:
            if not c.skip():
                total_force += np.array(c.getImpulse())
        return total_force / self.timestep