"""
Keyboard input handler with the same interface as the joystick handler.

Controls:
    UP or W    -> decrease force
    DOWN or S  -> increase force

Keep the pygame control window focused while driving the robot.
"""

import logging

import numpy as np
import pygame

from config import DEFAULT_CONFIG, HitlConfig
from joystick_handler import JoystickHandler

logger = logging.getLogger(__name__)


class KeyboardHandler(JoystickHandler):
    """Keyboard-driven correction source that reuses the joystick UI."""

    def __init__(self, config: HitlConfig = None):
        super().__init__(config=config)
        self.config = config or DEFAULT_CONFIG
        self.human_cfg = self.config.human_input
        self.quit_requested = False

        if self.display_enabled:
            pygame.display.set_caption("HITL Control - Keyboard")
            logger.info("Keyboard control ready. Focus the pygame window to capture key presses.")
        else:
            logger.warning("Keyboard control needs the pygame window to be available.")

    def _init_joystick(self):
        """Skip joystick discovery; keyboard input is handled through pygame."""
        self.joystick = None
        self.controller = None
        self.haptics_available = False
        logger.info("Using keyboard control")

    def _process_events(self):
        """Keep the pygame window responsive and detect close requests."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                self.display_enabled = False
                logger.info("Keyboard control window closed. Continuing without visual overlay.")

    def get_action(self, current_force_mag: float = 0.0) -> np.ndarray:
        """
        Return the current human correction as a normalized force delta.

        The pygame window must stay focused for key presses to register.
        """
        self._process_events()

        if not pygame.get_init() or not pygame.display.get_init():
            self.last_action_value = 0.0
            self.last_force_mag = current_force_mag
            return np.array([0.0], dtype=np.float32)

        pygame.event.pump()

        keys = pygame.key.get_pressed()
        action_val = 0.0

        # Up/W releases force, Down/S increases force.
        decrease_pressed = keys[pygame.K_UP] or keys[pygame.K_w]
        increase_pressed = keys[pygame.K_DOWN] or keys[pygame.K_s]

        if decrease_pressed and not increase_pressed:
            action_val = -self.human_cfg.sensitivity
        elif increase_pressed and not decrease_pressed:
            action_val = self.human_cfg.sensitivity

        self.last_action_value = float(np.clip(action_val, -1.0, 1.0))
        self.last_force_mag = current_force_mag

        if abs(self.last_action_value) > 0.01:
            self.intervention_count += 1

        self.action_history.append(self.last_action_value)

        self.frame_count += 1
        if self.terminal_output_enabled and self.frame_count % 20 == 0:
            self._print_terminal_feedback(current_force_mag)

        return np.array([self.last_action_value], dtype=np.float32)

    def _get_guidance(self, force_mag: float, human_input: float):
        """Show keyboard-specific guidance in the overlay."""
        guidance = []

        error = force_mag - 10.0

        if abs(human_input) > 0.01:
            guidance.append(("Keyboard input is LIVE - robot reacting!", (100, 255, 150)))
        else:
            guidance.append(("Press W/UP or S/DOWN to help the robot", (150, 200, 200)))

        if error > 3.0:
            guidance.append(("Press W or UP (force too high)", (255, 150, 100)))
        elif error < -3.0:
            guidance.append(("Press S or DOWN (force too low)", (255, 150, 100)))
        elif error > 1.0:
            guidance.append(("Gently press W or UP", (255, 200, 100)))
        elif error < -1.0:
            guidance.append(("Gently press S or DOWN", (255, 200, 100)))
        else:
            guidance.append(("Target force reached - hold steady", (100, 255, 150)))

        return guidance

    def close(self):
        """Clean up pygame resources for keyboard control."""
        if pygame.get_init():
            pygame.quit()
        logger.info("KeyboardHandler closed")
