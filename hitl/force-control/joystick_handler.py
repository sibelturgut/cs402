"""
Robust joystick input handler with visual feedback.
"""

import logging
from collections import deque

import numpy as np
import pygame

from config import HitlConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

try:
    import pygame._sdl2.controller as sdl2_controller
except Exception:
    sdl2_controller = None


class JoystickHandler:
    """
    Handles joystick input with visual feedback overlay and optional rumble.

    Input still comes from ``pygame.joystick`` because that path is already
    stable in this project. Haptics use SDL2 controller support when the same
    device is also recognized as a game controller.
    """

    def __init__(self, config: HitlConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.human_cfg = self.config.human_input

        pygame.init()
        pygame.joystick.init()

        self.joystick = None
        self.controller = None
        self.haptics_available = False
        self._last_rumble_sent_ms = 0
        self._last_rumble_state = (0.0, 0.0)
        self._was_on_target = False
        self._init_joystick()

        # Display
        self.display_enabled = True
        self._init_display()

        # Statistics
        self.frame_count = 0
        self.last_action_value = 0.0
        self.intervention_count = 0
        self.last_force_mag = 0.0
        self.action_history = deque(maxlen=5)  # Last 5 actions
        self.terminal_output_enabled = True
        self.human_accumulated_force = 0.0  # Track human's cumulative force
        self.cumulative_force_history = deque(maxlen=100)  # History for moving average
        self.ai_intent_force = 0.0
        self.autonomy_enabled = False

    def _normalize_axis(self, value: float) -> float:
        """Normalize raw controller axis values to ``[-1, 1]``."""
        value = float(value)
        # SDL controller axes are commonly reported as signed 16-bit ints.
        if abs(value) > 1.0:
            value /= 32767.0
        return float(np.clip(value, -1.0, 1.0))

    def _init_joystick(self):
        """Initialize joystick input and optional haptic support."""
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"Joystick detected: {self.joystick.get_name()}")
            self._init_haptics()
        else:
            logger.warning("No joystick detected! Simulation will run autonomously.")

    def _init_haptics(self):
        """Attach SDL2 haptics when the joystick is also a recognized controller."""
        if not self.human_cfg.enable_haptics:
            logger.info("Joystick haptics disabled in config")
            return

        if sdl2_controller is None:
            logger.warning("SDL2 controller support is unavailable; joystick rumble disabled")
            return

        try:
            sdl2_controller.init()
            if not sdl2_controller.is_controller(0):
                logger.warning("Joystick is not exposed as an SDL2 game controller; rumble disabled")
                return

            self.controller = sdl2_controller.Controller.from_joystick(self.joystick)
            self.haptics_available = self.controller is not None
            if self.haptics_available:
                logger.info("Joystick rumble support enabled")
        except Exception as exc:
            self.controller = None
            self.haptics_available = False
            logger.warning(f"Could not initialize joystick rumble: {exc}")

    def _read_controller_action(self) -> float | None:
        """Read PS4/Xbox-style input through SDL's controller mapping when available."""
        if self.controller is None:
            return None

        try:
            if hasattr(self.controller, "attached") and not self.controller.attached():
                return None

            # SDL standard controller button indices.
            dpad_up = bool(self.controller.get_button(11))
            dpad_down = bool(self.controller.get_button(12))
            if dpad_up and not dpad_down:
                return float(-self.human_cfg.sensitivity)
            if dpad_down and not dpad_up:
                return float(self.human_cfg.sensitivity)

            axis_y = self._normalize_axis(self.controller.get_axis(1))
            if abs(axis_y) > self.human_cfg.deadzone:
                # Up on the stick is negative axis and should decrease force.
                return float(axis_y * self.human_cfg.sensitivity)
        except Exception as exc:
            logger.warning(f"SDL controller read failed, falling back to raw joystick input: {exc}")
            self.controller = None
            self.haptics_available = False

        return None

    def _read_legacy_joystick_action(self) -> float:
        """Fallback input path for devices not exposed as SDL game controllers."""
        if self.joystick is None:
            return 0.0

        action_val = 0.0
        try:
            if hasattr(self.joystick, "get_numhats") and self.joystick.get_numhats() > 0:
                _, hat_y = self.joystick.get_hat(0)
                if hat_y != 0:
                    # Up should decrease force, matching keyboard mode.
                    return float(-hat_y * self.human_cfg.sensitivity)

            if hasattr(self.joystick, "get_numaxes") and self.joystick.get_numaxes() > 1:
                axis_y = self._normalize_axis(self.joystick.get_axis(1))
                if abs(axis_y) > self.human_cfg.deadzone:
                    return float(axis_y * self.human_cfg.sensitivity)
        except Exception as exc:
            logger.warning(f"Raw joystick read failed: {exc}")

        return action_val

    def _init_display(self):
        """Initialize pygame display for feedback"""
        try:
            self.screen = pygame.display.set_mode((500, 400))
            pygame.display.set_caption("HITL Control - Human in the Loop")
            self.font_large = pygame.font.SysFont(None, 28)
            self.font_normal = pygame.font.SysFont(None, 24)
            self.font_small = pygame.font.SysFont(None, 20)
        except Exception as e:
            logger.warning(f"Could not initialize display: {e}")
            self.display_enabled = False

    def get_action(self, current_force_mag: float = 0.0) -> np.ndarray:
        """
        Get human action from joystick as force delta (change command).

        Delta Control:
            UP    → action = -1.0 → decrease force by ~0.2N per step
            DOWN  → action = +1.0 → increase force by ~0.2N per step
            IDLE  → action = 0.0  → maintain current force (no change!)

        This way:
            - When you release joystick, force stays where you left it
            - You modulate UP/DOWN from there
            - Target: 10N (what the AI learns)

        Args:
            current_force_mag: Current measured force for UI feedback

        Returns:
            Normalized action in [-1, 1] representing force delta
        """
        pygame.event.pump()
        action_val = self._read_controller_action()
        if action_val is None:
            action_val = self._read_legacy_joystick_action()
        
        # Output is [-1, 1] representing force command
        self.last_action_value = float(np.clip(action_val, -1.0, 1.0))
        self.last_force_mag = current_force_mag

        # Track only local stats here; authoritative human force stored on the env
        if abs(self.last_action_value) > 0.01:
            self.intervention_count += 1

        self.action_history.append(self.last_action_value)

        # Terminal output occasionally (no heavy rendering here)
        self.frame_count += 1
        if self.terminal_output_enabled and self.frame_count % 20 == 0:
            self._print_terminal_feedback(current_force_mag)

        return np.array([self.last_action_value], dtype=np.float32)

    def _print_terminal_feedback(self, force_mag: float):
        """Print real-time feedback to terminal with force tracking"""
        error = force_mag - 10.0

        # Force status
        if abs(error) < 1.0:
            status = "✓ GOOD"
        elif abs(error) < 3.0:
            status = "OK"
        else:
            status = "ERROR"

        # Direction hint
        direction = ""
        if error > 1.0:
            direction = "↑UP to decrease"
        elif error < -1.0:
            direction = "↓DOWN to increase"

        # Your input
        if abs(self.last_action_value) > 0.05:
            input_status = f"YOU: {self.last_action_value:+.3f}"
        else:
            input_status = "YOU: idle"

        # Human force contribution
        human_force_display = f"Your Contrib: {self.human_accumulated_force:+.2f}N"

        logger.info(
            f"{input_status:20s} | "
            f"Actual: {force_mag:+06.2f}N | "
            f"Target: 10.00N | "
            f"Error: {error:+.2f}N | "
            f"{status:7s} | "
            f"{human_force_display:20s} | "
            f"{direction}"
        )

    def _interpolate_rumble_strength(self, abs_error: float) -> float:
        """Map force error to Gaussian-decayed off-target rumble.

        The strongest continuous rumble is just outside the target band. As the
        force error grows, rumble decays smoothly toward a low background floor.
        """
        band_edge = self.human_cfg.off_target_min_error
        falloff_horizon = max(band_edge + 1e-6, self.human_cfg.off_target_max_error)
        floor_rumble = self.human_cfg.off_target_min_rumble
        peak_rumble = self.human_cfg.off_target_max_rumble

        if abs_error <= band_edge:
            return float(peak_rumble)

        # Treat `off_target_max_error` as the point where the Gaussian is nearly
        # flat at the floor value (about 3 sigma from the band edge).
        sigma = max((falloff_horizon - band_edge) / 3.0, 1e-3)
        distance_from_band = abs_error - band_edge
        gaussian = float(np.exp(-0.5 * (distance_from_band / sigma) ** 2))
        strength = floor_rumble + (peak_rumble - floor_rumble) * gaussian
        return float(np.clip(strength, floor_rumble, peak_rumble))

    def _set_rumble(self, low: float, high: float, duration_ms: int) -> None:
        """Send a rumble command only when a haptic-capable controller is available."""
        if not self.haptics_available or self.controller is None:
            return

        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        now_ms = pygame.time.get_ticks()
        state_changed = abs(low - self._last_rumble_state[0]) > 0.02 or abs(high - self._last_rumble_state[1]) > 0.02
        refresh_due = now_ms - self._last_rumble_sent_ms >= self.human_cfg.rumble_refresh_ms
        if not state_changed and not refresh_due:
            return

        try:
            if low == 0.0 and high == 0.0:
                self.controller.stop_rumble()
            else:
                self.controller.rumble(low, high, duration_ms)
            self._last_rumble_state = (low, high)
            self._last_rumble_sent_ms = now_ms
        except Exception as exc:
            logger.warning(f"Joystick rumble update failed: {exc}")
            self.haptics_available = False

    def _update_haptics(self, measured_force: float) -> None:
        """Pulse on target acquisition and apply graded rumble while off target."""
        if not self.haptics_available:
            return

        error = measured_force - self.config.robot.target_force_mag
        abs_error = abs(error)

        if abs_error <= self.human_cfg.on_target_band:
            # Emit one short confirmation pulse when re-entering the target band.
            if not self._was_on_target:
                self._set_rumble(
                    low=self.human_cfg.on_target_pulse_low,
                    high=self.human_cfg.on_target_pulse_high,
                    duration_ms=self.human_cfg.on_target_pulse_ms,
                )
            else:
                self._set_rumble(0.0, 0.0, self.human_cfg.rumble_refresh_ms)
            self._was_on_target = True
            return

        self._was_on_target = False
        strength = self._interpolate_rumble_strength(abs_error)
        if strength <= 0.0:
            self._set_rumble(0.0, 0.0, self.human_cfg.rumble_refresh_ms)
            return

        # Bias low vs high frequency so the operator can feel which side of the
        # target they are on without looking at the screen.
        if error < 0.0:
            low, high = strength, strength * 0.35
        else:
            low, high = strength * 0.35, strength

        self._set_rumble(low, high, self.human_cfg.rumble_refresh_ms + 30)

    def update_display(self, measured_force: float):
        """Update haptics and render the latest measured and intent forces."""
        try:
            self.last_force_mag = measured_force
            self._update_haptics(measured_force)
            if self.display_enabled:
                self._render_feedback(measured_force)
        except Exception as e:
            logger.warning(f"Update display failed: {e}")

    def _render_feedback(self, measured_force: float):
        """Render real-time visual feedback with separate measured/human/AI meters"""
        try:
            self.screen.fill((15, 15, 25))

            # Title
            title = self.font_large.render("PANDA REAL-TIME CONTROL", True, (255, 215, 0))
            self.screen.blit(title, (40, 8))

            # Measured force (big)
            force_error = abs(measured_force - self.config.robot.target_force_mag)
            force_color = (100, 255, 100) if force_error < 2.0 else (255, 150, 100)
            force_display = self.font_large.render(f"Measured: {measured_force:+.2f} N", True, force_color)
            self.screen.blit(force_display, (40, 44))

            # Draw measured force meter (0..20N)
            self._draw_measured_meter(measured_force, 90)

            # Commanded contribution stack: human vs AI intent
            human_mag = abs(self.human_accumulated_force)
            ai_mag = abs(self.ai_intent_force)
            total_cmd = human_mag + ai_mag
            cmd_text = self.font_normal.render(
                f"Human: {self.human_accumulated_force:+.2f}N   |   AI Intent: {self.ai_intent_force:+.2f}N   |   Total Cmd: {(-total_cmd):+.2f}N",
                True, (220, 220, 220)
            )
            self.screen.blit(cmd_text, (40, 170))

            # Draw stacked bar of contributions (magnitudes)
            bar_x, bar_y = 40, 200
            bar_w, bar_h = 420, 30
            pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_w, bar_h), 2)

            max_display = 20.0
            human_w = int((human_mag / max_display) * bar_w)
            ai_w = int((ai_mag / max_display) * bar_w)

            # Human portion (left)
            pygame.draw.rect(self.screen, (255, 150, 100), (bar_x, bar_y, human_w, bar_h))
            # AI portion stacked after human
            pygame.draw.rect(self.screen, (100, 150, 255), (bar_x + human_w, bar_y, ai_w, bar_h))

            # Target marker (10N)
            target_x = int(bar_x + (self.config.robot.target_force_mag / max_display) * bar_w)
            pygame.draw.line(self.screen, (200, 255, 100), (target_x, bar_y - 6), (target_x, bar_y + bar_h + 6), 2)

            # Labels
            left_label = self.font_small.render("0N", True, (150, 150, 150))
            right_label = self.font_small.render("20N", True, (150, 150, 150))
            self.screen.blit(left_label, (bar_x - 8, bar_y + bar_h + 6))
            self.screen.blit(right_label, (bar_x + bar_w - 28, bar_y + bar_h + 6))

            # Status
            if abs(self.last_action_value) > 0.01:
                status = "🎮 YOU ARE CONTROLLING"
                status_color = (100, 255, 150)
            else:
                if self.autonomy_enabled:
                    status = "🤖 AUTONOMY ENABLED"
                    status_color = (100, 200, 255)
                else:
                    status = "🤖 AI (waiting for guidance)"
                    status_color = (100, 150, 255)

            status_text = self.font_normal.render(status, True, status_color)
            self.screen.blit(status_text, (40, 246))

            # Guidance
            guidance = self._get_guidance(measured_force, self.last_action_value)
            guidance_y = 285
            for line, color in guidance:
                text = self.font_small.render(line, True, color)
                self.screen.blit(text, (40, guidance_y))
                guidance_y += 20

            pygame.display.flip()
        except Exception as e:
            logger.warning(f"Render error: {e}")

    def _draw_force_meter(self, current_force: float, y_pos: int):
        """Legacy meter kept for compatibility; not used by new UI."""
        return

    def _draw_measured_meter(self, measured_force: float, y_pos: int):
        """Draw measured force meter with 0..20N range and target zone 8-12N"""
        min_force = 0.0
        max_force = 20.0
        bar_x, bar_y = 40, y_pos
        bar_w, bar_h = 420, 30

        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_w, bar_h), 2)

        # Good zone (8-12N)
        good_min = int(bar_x + (8.0 - min_force) / (max_force - min_force) * bar_w)
        good_max = int(bar_x + (12.0 - min_force) / (max_force - min_force) * bar_w)
        pygame.draw.rect(self.screen, (50, 150, 50), (good_min, bar_y, max(1, good_max - good_min), bar_h))

        # Current force position
        norm = np.clip((measured_force - min_force) / (max_force - min_force), 0.0, 1.0)
        pos_x = int(bar_x + norm * bar_w)
        color = (100, 255, 100) if abs(measured_force - 10.0) < 2.0 else (255, 150, 100)
        pygame.draw.circle(self.screen, color, (pos_x, bar_y + bar_h // 2), 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (pos_x, bar_y + bar_h // 2), 10, 2)

        # Labels
        left_label = self.font_small.render("0N", True, (150, 150, 150))
        right_label = self.font_small.render("20N", True, (150, 150, 150))
        target_label = self.font_small.render("10N", True, (200, 255, 100))
        self.screen.blit(left_label, (bar_x - 8, bar_y + bar_h + 6))
        self.screen.blit(right_label, (bar_x + bar_w - 28, bar_y + bar_h + 6))
        target_x = int(bar_x + (10.0 - min_force) / (max_force - min_force) * bar_w)
        self.screen.blit(target_label, (target_x - 18, bar_y - 18))

    def _get_guidance(self, force_mag: float, human_input: float):
        """Get real-time guidance on what to do"""
        guidance = []

        error = force_mag - 10.0

        if abs(human_input) > 0.01:
            guidance.append(("✓ Your input is LIVE - robot reacting!", (100, 255, 150)))
        else:
            guidance.append(("← Move D-Pad to help the robot →", (150, 200, 200)))

        if error > 3.0:
            guidance.append(("PUSH UP ↑ (force too high)", (255, 150, 100)))
        elif error < -3.0:
            guidance.append(("PUSH DOWN ↓ (force too low)", (255, 150, 100)))
        elif error > 1.0:
            guidance.append(("Gently push UP ↑", (255, 200, 100)))
        elif error < -1.0:
            guidance.append(("Gently push DOWN ↓", (255, 200, 100)))
        else:
            guidance.append(("✓ PERFECT! Hold steady", (100, 255, 150)))

        if self.haptics_available:
            guidance.append(("Rumble: pulse on target, graded feedback off target", (150, 200, 200)))

        return guidance

    def reset_stats(self):
        """Reset statistics for new episode"""
        self.intervention_count = 0
        self.human_accumulated_force = 0.0  # Reset human force tracking
        self.cumulative_force_history.clear()

    def close(self):
        """Clean up pygame resources"""
        if self.controller is not None:
            try:
                self.controller.stop_rumble()
            except Exception:
                pass
            try:
                self.controller.quit()
            except Exception:
                pass
        if sdl2_controller is not None:
            try:
                sdl2_controller.quit()
            except Exception:
                pass
        if self.display_enabled:
            pygame.quit()
        logger.info("JoystickHandler closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = JoystickHandler()
    
    for i in range(100):
        action = handler.get_action(current_force_mag=10.0 + np.random.randn() * 0.5)
        if i % 10 == 0:
            print(f"Frame {i}: Action={action[0]:.3f}")
    
    handler.close()
