import pygame
import numpy as np

class JoystickController:
    def __init__(self, sensitivity=0.1): # Lowered base sensitivity for "Small Actions"
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Force Control - Precision Mode")
        
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 48)
        self.target_force = 10.0  
        self.base_sensitivity = sensitivity

    def get_action(self, current_force_mag=0.0):
        pygame.event.pump()
        action_val = 0.0
        is_fine_control = False

        if self.joystick:
            hat_x, hat_y = self.joystick.get_hat(0)
            
            # Use 'Cross' (0) for EXTRA fine control
            if self.joystick.get_button(0):
                is_fine_control = True
            
            current_sens = self.base_sensitivity
            if is_fine_control:
                current_sens *= 0.2 # Extremely slow
            
            # Map D-Pad to action
            if hat_y == 1:    # UP -> Move Up (Decrease Force)
                action_val = -current_sens
            elif hat_y == -1: # DOWN -> Move Down (Increase Force)
                action_val = current_sens
            
        self._render_ui(current_force_mag, is_fine_control)
        return np.array([action_val], dtype=np.float32)

    def _render_ui(self, force_mag, is_fine_control):
        # ... (Same UI code as before) ...
        self.screen.fill((30, 30, 30))
        error = abs(force_mag - self.target_force)
        color = (50, 255, 50) if error <= 0.5 else (255, 80, 80)
        force_text = self.large_font.render(f"{force_mag:.2f} N", True, color)
        self.screen.blit(force_text, (135, 60))
        
        status = "SLOW (In Air)" if force_mag < 0.1 else "NORMAL (Contact)"
        if is_fine_control: status = "ULTRA-PRECISION"
        
        txt = self.font.render(f"STATUS: {status}", True, (200, 255, 200))
        self.screen.blit(txt, (100, 20))
        pygame.display.flip()