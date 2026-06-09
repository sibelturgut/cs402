import pygame
import numpy as np

class KeyboardController:
    def __init__(self, sensitivity=0.1): # <--- TUNE THIS VALUE
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Live Force Dashboard")
        
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 48)
        self.target_force = 10.0  
        
        # This controls how fast the force changes per physics step
        self.base_sensitivity = sensitivity 

    def get_action(self, current_force_mag=0.0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
        keys = pygame.key.get_pressed()
        action_val = 0.0
        
        # --- Fine Control Modifier ---
        # Hold Left or Right SHIFT to move 10x slower for precision
        current_sensitivity = self.base_sensitivity
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            current_sensitivity = self.base_sensitivity * 0.1 
        
        # Apply the scaled action
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action_val = -current_sensitivity  
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action_val = current_sensitivity   
            
        self._render_ui(current_force_mag, (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]))
            
        return np.array([action_val], dtype=np.float32)

    def _render_ui(self, force_mag, is_fine_control):
        self.screen.fill((30, 30, 30))
        
        error = abs(force_mag - self.target_force)
        if error <= 0.5:
            color = (50, 255, 50)   # Green
        elif error <= 2.0:
            color = (255, 200, 50)  # Yellow
        else:
            color = (255, 80, 80)   # Red
            
        force_text = self.large_font.render(f"{force_mag:.2f} N", True, color)
        self.screen.blit(force_text, (135, 60))
        
        # UI Updates to show controls
        text1 = self.font.render("W: Push Down  |  S: Ease Off", True, (200, 200, 200))
        self.screen.blit(text1, (85, 10))
        
        # Show if fine control is active
        shift_color = (100, 255, 255) if is_fine_control else (100, 100, 100)
        text2 = self.font.render("[HOLD SHIFT FOR FINE CONTROL]", True, shift_color)
        self.screen.blit(text2, (65, 30))

        # --- Gauge Drawing ---
        bar_width = 300
        bar_height = 40
        bar_x = 50
        bar_y = 150
        
        pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
        fill_width = min(int((force_mag / 20.0) * bar_width), bar_width)
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
        
        target_x = bar_x + int((self.target_force / 20.0) * bar_width)
        pygame.draw.line(self.screen, (255, 255, 255), (target_x, bar_y - 15), (target_x, bar_y + bar_height + 15), 4)
        
        target_text = self.font.render("Target (10N)", True, (255, 255, 255))
        self.screen.blit(target_text, (target_x - 45, bar_y + bar_height + 20))
        
        pygame.display.flip()