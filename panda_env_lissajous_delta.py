import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import raisimpy as raisim

class PandaEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.timestep = 0.01 
        
        # 1D ACTION SPACE: AI outputs [-1, 1] which we dynamically scale into a delta force
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        
        # 20D OBSERVATION SPACE: q(7) + v(7) + force(3) + dist(1) + current_z_force(1) + error_sum(1) = 20
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float64)
        
        # --- RAISIM SETUP ---
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
        self.world = raisim.World()
        self.world.addGround()
        self.world.setTimeStep(self.timestep)
        
        self.panda = self.world.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/panda/urdf/panda.urdf")
        self.panda.setName("panda")

        # --- BOX SETUP ---
        self.box_center_x = 0.2
        self.box_center_y = -0.45
        self.box_pos = [self.box_center_x, self.box_center_y, 0.1]
        self.box = self.world.addBox(0.4, 0.4, 0.5, 100.0)
        self.box.setPosition(*self.box_pos)
        self.box.setBodyType(raisim.BodyType.STATIC)

        # --- END EFFECTOR & TARGETS ---
        self.target_force_mag = 10.0 
        self.ee_frame_idx = self.panda.getFrameIdxByName("EndEffector_Sphere")
        self.ee_frame_name = "EndEffector_Sphere"
        self.ee_offset = 0.014142 
        
        self.t = 0.0

        # Safe starting joint configuration
        self.contact_q = np.array(([-1.03996814, -0.24488271, -0.13182964, -2.39703664,  0.13306204,
        2.33681407,  0.69235052]))

    def _get_lissajous_target(self):
        """Generates a gentle Figure-8 that starts EXACTLY at the center."""
        f_x = 1.0 / 10.0  
        f_y = 2.0 / 10.0  
        A = 0.06  
        B = 0.06  
        
        # REMOVED delta so sin(0) = 0 at the start
        target_x = self.box_center_x + A * np.sin(2.0 * np.pi * f_x * self.t)
        target_y = self.box_center_y + B * np.sin(2.0 * np.pi * f_y * self.t)
        
        return np.array([target_x, target_y])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0 
        
        # NEW: Initialize state variables for precision control
        self.current_force_z = 0.0 
        self.error_sum = 0.0          # Integral error accumulator
        
        self.panda.setGeneralizedCoordinate(self.contact_q)
        self.panda.setGeneralizedVelocity(np.zeros(7))
        self.panda.setGeneralizedForce(np.zeros(7))
        return self._get_obs(), {}

    def step(self, action):
        self.t += self.timestep
        
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        ee_vel = self.panda.getFrameVelocity(self.ee_frame_idx) 
        
        # ==========================================
        # 1. THE VIRTUAL SPRING (Math controls X/Y)
        # ==========================================
        target_xy = self._get_lissajous_target()
        Kp = 800.0
        Kd = 30.0
        
        force_x = (Kp * (target_xy[0] - ee_pos[0])) - (Kd * ee_vel[0])
        force_y = (Kp * (target_xy[1] - ee_pos[1])) - (Kd * ee_vel[1])
        
        # ==========================================
        # 2. THE AI (Consistent Delta + Z-Damping)
        # ==========================================
        raw_action = float(action[0])
        
        # Consistent small delta (max 0.25N change per step)
        delta_force = raw_action * 0.25
        
        # Apply delta and clip for safety [-20N to -2N]
        self.current_force_z = np.clip(self.current_force_z + delta_force, -20.0, -2.0)
        
        # NEW: Physical Z-Damping! This acts as a shock absorber against the box
        Kd_z = 25.0
        damping_z = -Kd_z * ee_vel[2]
        
        final_force_z = self.current_force_z + damping_z
        desired_force_at_ee = np.array([force_x, force_y, final_force_z]) 
        
        # ==========================================
        # 3. PHYSICS EXECUTION
        # ==========================================
        J = self.panda.getDenseFrameJacobian(self.ee_frame_name)
        tau_action = np.matmul(J[:3, :].T, desired_force_at_ee)
        tau_grav = self.panda.getNonlinearities(self.world.getGravity())
        tau_damping = -1.0 * self.panda.getGeneralizedVelocity() 

        self.panda.setGeneralizedForce(tau_action + tau_grav + tau_damping)
        self.world.integrate()
        
        # ==========================================
        # 4. REWARD CALCULATION
        # ==========================================
        new_ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist_to_surface = max(0, new_ee_pos[2] - 0.35 - self.ee_offset)
        
        new_force = self._get_contact_force()
        new_force_mag = np.linalg.norm(new_force)
        force_error = abs(new_force_mag - self.target_force_mag)

        # Keep integral error for observation, but clip it tightly
        signed_error = new_force_mag - self.target_force_mag
        self.error_sum += signed_error * self.timestep
        self.error_sum = np.clip(self.error_sum, -2.0, 2.0)

        if dist_to_surface > 0.002:
            reward = -50.0 * dist_to_surface 
        else:
            # Steep exponential reward: very smooth gradient, but spikes high near 0 error
            # We also heavily penalize large actions so it learns to output 0.0 when on target
            action_penalty = 1.5 * abs(raw_action)
            reward = 10.0 * np.exp(-2.5 * force_error) - action_penalty
            
        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        q = self.panda.getGeneralizedCoordinate()
        v = self.panda.getGeneralizedVelocity()
        f = self._get_contact_force()
        ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
        dist = max(0, ee_pos[2] - 0.35 - self.ee_offset)
        
        # Add the agent's current Z-force and the integral error to the observation
        return np.concatenate([q, v, f, [dist], [self.current_force_z], [self.error_sum]])

    def _get_contact_force(self):
        contacts = self.panda.getContacts()
        total_force = np.zeros(3)
        for c in contacts:
            if not c.skip():
                total_force += np.array(c.getImpulse())
        return total_force / self.timestep
    
#     def calibrate_perfect_start(self):
#         print("\n--- STARTING CALIBRATION ---")
#         print("Finding the perfect center joints. Please wait 3 seconds...\n")
        
#         # Reset to a default state first
#         self.panda.setGeneralizedCoordinate(self.contact_q)
#         self.world.integrate()

#         # Give it 300 steps (3 seconds) to settle perfectly
#         for i in range(300):
#             ee_pos = self.panda.getFramePosition(self.ee_frame_idx)
#             ee_vel = self.panda.getFrameVelocity(self.ee_frame_idx)
            
#             # Target the exact center of the box
#             target_x = self.box_center_x
#             target_y = self.box_center_y
            
#             # MASSIVE gains to force it to the center against any friction
#             Kp = 3000.0
#             Kd = 150.0
            
#             force_x = (Kp * (target_x - ee_pos[0])) - (Kd * ee_vel[0])
#             force_y = (Kp * (target_y - ee_pos[1])) - (Kd * ee_vel[1])
            
#             # Push down hard enough to stay touching the box
#             force_z = -20.0 
            
#             desired_force_at_ee = np.array([force_x, force_y, force_z]) 
#             J = self.panda.getDenseFrameJacobian(self.ee_frame_name)
#             tau_action = np.matmul(J[:3, :].T, desired_force_at_ee)
            
#             tau_grav = self.panda.getNonlinearities(self.world.getGravity())
#             # Heavy joint damping so it doesn't vibrate
#             tau_damping = -5.0 * self.panda.getGeneralizedVelocity() 

#             self.panda.setGeneralizedForce(tau_action + tau_grav + tau_damping)
#             self.world.integrate()

#         print("CALIBRATION COMPLETE! 🎯")
#         print("Copy the array below and paste it into your __init__ for self.contact_q:")
#         print("-" * 50)
#         print(repr(self.panda.getGeneralizedCoordinate()))
#         print("-" * 50)
#         exit() # Stop the program so you can copy the numbers

# if __name__ == "__main__":
#     env = PandaEnv()
#     env.calibrate_perfect_start()
