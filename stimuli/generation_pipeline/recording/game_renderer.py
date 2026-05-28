import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import pygame
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import config for game constants with proper working directory
original_cwd = os.getcwd()
try:
    # Change to parent directory where assets are located
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)
    
    from core import config
finally:
    # Restore original working directory
    os.chdir(original_cwd)

class GameRenderer:
    """Renders game frames from saved recording data"""
    
    def __init__(self, bird_sprite_name='bird-flap-1.png', hide_ui=False):
        # Initialize pygame
        pygame.init()
        
        # Store custom bird sprite name
        self.bird_sprite_name = bird_sprite_name
        
        # Store UI visibility preference
        self.hide_ui = hide_ui
        
        # Create rendering surface
        self.surface = pygame.Surface((config.SCREEN_WIDHT, config.SCREEN_HEIGHT))
        
        # Load game assets
        self.load_assets()
        
        # Track flap animation state across frames
        self.flap_animation_state = {}  # Dict mapping frame index to animation frame
        
        # Colors for drawing
        self.colors = {
            'background': (135, 206, 235),  # Sky blue
            'bird': (255, 215, 0),          # Gold
            'pipe': (0, 128, 0),            # Green
            'ground': (222, 184, 135),      # Burlywood
            'text': (255, 255, 255),        # White
            'black': (0, 0, 0),
            'grey': (80, 80, 80)
        }
    
    def load_assets(self):
        """Load game assets for rendering"""
        try:
            # Try to load original assets with proper paths
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
            
            # Use custom bird sprite (allows different models to have different bird colors)
            self.bird_image = pygame.image.load(os.path.join(assets_dir, self.bird_sprite_name))
            self.bird_image = pygame.transform.scale(self.bird_image, (config.BIRD_WIDTH, config.BIRD_HEIGHT))
            
            # Load flap animation sprites that match the selected bird family.
            try:
                flap1_name, flap2_name = self._flap_sprite_names(self.bird_sprite_name)
                self.bird_flap1_image = pygame.image.load(os.path.join(assets_dir, flap1_name))
                self.bird_flap1_image = pygame.transform.scale(self.bird_flap1_image, (config.BIRD_WIDTH, config.BIRD_HEIGHT))
                
                self.bird_flap2_image = pygame.image.load(os.path.join(assets_dir, flap2_name))
                self.bird_flap2_image = pygame.transform.scale(self.bird_flap2_image, (config.BIRD_WIDTH, config.BIRD_HEIGHT))
            except Exception as e:
                # Fallback to main bird image if flap sprites don't exist
                print(f"Warning: Could not load flap animation sprites: {e}")
                self.bird_flap1_image = self.bird_image
                self.bird_flap2_image = self.bird_image

            self.pipe_image = pygame.image.load(os.path.join(assets_dir, 'pipe-green.png'))
            self.pipe_image = pygame.transform.scale(self.pipe_image, (config.PIPE_WIDHT, config.PIPE_HEIGHT))
            
            self.ground_image = pygame.image.load(os.path.join(assets_dir, 'base.png'))
            self.ground_image = pygame.transform.scale(self.ground_image, (config.GROUND_WIDHT, config.GROUND_HEIGHT))
            
            self.background_image = pygame.image.load(os.path.join(assets_dir, 'background-day.png'))
            self.background_image = pygame.transform.scale(self.background_image, (config.SCREEN_WIDHT, config.SCREEN_HEIGHT))
            
            self.goal_image = pygame.image.load(os.path.join(assets_dir, 'goal.png'))
            self.goal_image = pygame.transform.scale(self.goal_image, (config.PIPE_WIDHT, config.PIPE_GAP))
            
            # Also create PIL version for PIL-based rendering
            self.goal_image_pil = Image.open(os.path.join(assets_dir, 'goal.png'))
            self.goal_image_pil = self.goal_image_pil.resize((config.PIPE_WIDHT, config.PIPE_GAP), Image.LANCZOS)
            
            self.has_assets = True
            
        except Exception as e:
            print(f"Could not load assets: {e}")
            print("Using simple shapes for rendering")
            self.has_assets = False

    @staticmethod
    def _flap_sprite_names(sprite_name):
        """Return the two animated flap frames for a bird sprite filename."""
        if 'upflap' in sprite_name:
            return sprite_name.replace('upflap', 'flap'), sprite_name.replace('upflap', 'flap2')
        if 'flap2' in sprite_name:
            return sprite_name.replace('flap2', 'flap'), sprite_name
        if 'flap' in sprite_name:
            return sprite_name, sprite_name.replace('flap', 'flap2')
        return sprite_name, sprite_name
    
    def get_bird_sprite_for_frame(self, frame_data, episode_data=None):
        """Determine which bird sprite to use based on flap animation state"""
        # Get current step and action
        current_step = frame_data.get('step', 0)
        action_executed = frame_data.get('action_executed', 0)
        
        # Check if bird just flapped (action_executed == 1)
        if action_executed == 1:
            # Start flap animation at this step
            self.flap_animation_state[current_step] = 0
        
        # Determine animation frame based on recent flap history
        # Animation cycle: upflap -> flap1 (2 frames) -> flap2 (2 frames) -> upflap
        # Total animation duration: 4 frames after the jump
        animation_frame = 0
        
        # Look back up to 6 steps to find if we're in a flap animation
        for i in range(6):
            check_step = current_step - i
            if check_step in self.flap_animation_state:
                # We're i steps after a flap
                if i == 0 or i == 1:
                    animation_frame = 1  # Frames 0-1 after flap: flap1 (2 frames)
                elif i == 2 or i == 3:
                    animation_frame = 2  # Frames 2-3 after flap: flap2 (2 frames)
                else:
                    animation_frame = 0  # Back to normal upflap
                break
        
        # Select sprite based on animation frame
        if animation_frame == 1:
            return self.bird_flap1_image
        elif animation_frame == 2:
            return self.bird_flap2_image
        else:
            return self.bird_image
    
    def render_frame(self, frame_data):
        """Render a game frame from frame data"""
        if self.has_assets:
            return self._render_with_assets(frame_data)
        else:
            return self._render_simple(frame_data)
    
    def _render_with_assets(self, frame_data):
        """Render frame using loaded assets"""
        # Clear surface with background
        self.surface.blit(self.background_image, (0, 0))
        
        # Check if this frame comes from a damage-mode recording.
        damage_mode = frame_data.get('damage_mode', False)
        
        # Get bird data
        bird_data = frame_data.get('bird', {})
        bird_x = int(bird_data.get('x', 0))
        bird_y = int(bird_data.get('y', 0))
        
        # Use animated flap sprite based on recent actions.
        bird_sprite = self.get_bird_sprite_for_frame(frame_data)
        
        # In damage mode, draw bird on top; otherwise draw in original order
        if not damage_mode:
            # Normal mode: bird first
            self.surface.blit(bird_sprite, (bird_x, bird_y))
        
        # Draw pipes using actual recorded pipe positions (not regenerated from seed)
        # This ensures correct rendering even if the trajectory file has incorrect seed metadata
        pipes_data = frame_data.get('pipes', [])
        for pipe_data in pipes_data:
            pipe_x = int(pipe_data.get('x', 0))
            pipe_y = int(pipe_data.get('y', 0))
            inverted = pipe_data.get('inverted', False)
            
            # Create pipe image (flip if inverted)
            pipe_img = self.pipe_image.copy()
            if inverted:
                pipe_img = pygame.transform.flip(pipe_img, False, True)
            
            self.surface.blit(pipe_img, (pipe_x, pipe_y))
        
        # Draw ground
        ground_data = frame_data.get('ground', {})
        ground_x = int(ground_data.get('x', 0))
        ground_y = int(ground_data.get('y', config.SCREEN_HEIGHT - config.GROUND_HEIGHT))
        self.surface.blit(self.ground_image, (ground_x, ground_y))
        
        # Draw goal finish line in damage mode (if visible)
        if damage_mode:
            goal_x = frame_data.get('goal_x', None)
            goal_y = frame_data.get('goal_y', None)
            if goal_x is not None and goal_y is not None and -config.PIPE_WIDHT < goal_x < config.SCREEN_WIDHT:
                self.surface.blit(self.goal_image, (int(goal_x), int(goal_y)))

        # In damage mode, draw bird on top of everything (after pipes and ground)
        if damage_mode:
            self.surface.blit(bird_sprite, (bird_x, bird_y))
        
        # Draw score (or pipes in damage mode)
        font = pygame.font.Font(None, 36)
        
        if damage_mode:
            # In damage mode, show progress bar and damage icons below
            pipes_passed = frame_data.get('pipes_passed', 0)
            damage_taken = frame_data.get('damage_taken', 0)
            damage_target_pipes = frame_data.get('damage_target_pipes', 10)
            
            # Calculate continuous progress based on steps/frames
            # First pipe starts at X=280, bird left edge at X=50, bird width = 40
            # Distance for bird to fully clear first pipe: 280 - (50 + 40) = 190 pixels
            # Distance for remaining pipes: (target - 1) * 280 pixels
            
            step = frame_data.get('step', 0)
            game_speed = 5  # pixels per frame
            first_pipe_pos = 280  # first pipe X position
            bird_pos = 50  # bird X position (fixed)
            bird_width = 40  # bird width
            pipe_distance = 280  # pixels between pipes
            
            # Total distance = distance to first pipe + remaining pipes - bird width
            # (bird needs to fully clear, not just have left edge pass)
            total_distance = (first_pipe_pos - bird_pos) + ((damage_target_pipes - 1) * pipe_distance) - bird_width
            
            # Calculate progress as fraction of total distance traveled
            distance_traveled = 2 * step * game_speed
            continuous_progress = (distance_traveled / total_distance) * damage_target_pipes
            
            # Cap at target
            continuous_progress = min(continuous_progress, float(damage_target_pipes))
            
        else:
            # Normal mode, show score (unless UI is hidden)
            if not self.hide_ui:
                score = frame_data.get('score', 0)
                score_text = font.render(f"Score: {score}", True, self.colors['text'])
                self.surface.blit(score_text, (10, 10))
        
        # Draw step number (only in normal mode, not in damage mode, unless UI is hidden)
        if not damage_mode and not self.hide_ui:
            step = frame_data.get('step', 0)
            step_text = font.render(f"Step: {step}", True, self.colors['text'])
            self.surface.blit(step_text, (10, 50))
        # Convert pygame surface to PIL Image
        img_str = pygame.image.tostring(self.surface, 'RGB')
        pil_image = Image.frombytes('RGB', (config.SCREEN_WIDHT, config.SCREEN_HEIGHT), img_str)
        
        return pil_image
    
    def _render_simple(self, frame_data):
        """Render frame using simple shapes (fallback when assets not available)"""
        # Create PIL image
        img = Image.new('RGB', (config.SCREEN_WIDHT, config.SCREEN_HEIGHT), self.colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Check if damage mode
        damage_mode = frame_data.get('damage_mode', False)
        
        # Get bird data
        bird_data = frame_data.get('bird', {})
        bird_x = int(bird_data.get('x', 0))
        bird_y = int(bird_data.get('y', 0))
        bird_width = int(bird_data.get('width', config.BIRD_WIDTH))
        bird_height = int(bird_data.get('height', config.BIRD_HEIGHT))
        
        # In normal mode, draw bird first; in damage mode, draw bird last (on top)
        if not damage_mode:
            draw.ellipse(
                [bird_x, bird_y, bird_x + bird_width, bird_y + bird_height],
                fill=self.colors['bird'],
                outline=self.colors['black']
            )
        
        # Draw pipes as rectangles
        pipes_data = frame_data.get('pipes', [])
        for pipe_data in pipes_data:
            pipe_x = int(pipe_data.get('x', 0))
            pipe_y = int(pipe_data.get('y', 0))
            pipe_width = int(pipe_data.get('width', config.PIPE_WIDHT))
            pipe_height = int(pipe_data.get('height', config.PIPE_HEIGHT))
            
            draw.rectangle(
                [pipe_x, pipe_y, pipe_x + pipe_width, pipe_y + pipe_height],
                fill=self.colors['pipe'],
                outline=self.colors['black']
            )
        
        # Draw ground as rectangle
        ground_data = frame_data.get('ground', {})
        ground_x = int(ground_data.get('x', 0))
        ground_y = int(ground_data.get('y', config.SCREEN_HEIGHT - config.GROUND_HEIGHT))
        
        draw.rectangle(
            [ground_x, ground_y, ground_x + config.GROUND_WIDHT, ground_y + config.GROUND_HEIGHT],
            fill=self.colors['ground'],
            outline=self.colors['black']
        )
        
        # Draw goal finish line in damage mode (if visible)
        if damage_mode:
            goal_x = frame_data.get('goal_x', None)
            goal_y = frame_data.get('goal_y', None)
            if goal_x is not None and goal_y is not None and -config.PIPE_WIDHT < goal_x < config.SCREEN_WIDHT:
                # If we have the PIL asset, use it; otherwise draw a simple rectangle
                if self.has_assets and hasattr(self, 'goal_image_pil'):
                    if self.goal_image_pil.mode == 'RGBA':
                        img.paste(self.goal_image_pil, (int(goal_x), int(goal_y)), self.goal_image_pil)
                    else:
                        img.paste(self.goal_image_pil, (int(goal_x), int(goal_y)))
                else:
                    # Fallback: draw checkered pattern for finish line
                    draw.rectangle(
                        [int(goal_x), int(goal_y), int(goal_x) + config.PIPE_WIDHT, int(goal_y) + config.PIPE_GAP],
                        fill=(255, 255, 255),
                        outline=self.colors['black']
                    )
        
        # In damage mode, draw bird on top of everything
        if damage_mode:
            draw.ellipse(
                [bird_x, bird_y, bird_x + bird_width, bird_y + bird_height],
                fill=self.colors['bird'],
                outline=self.colors['black']
            )
        
        # Draw score and step text (or pipes/damage in damage mode)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        step = frame_data.get('step', 0)
        
        if damage_mode:
            # In damage mode, show progress bar and damage icons below (no step number)
            pipes_passed = frame_data.get('pipes_passed', 0)
            damage_taken = frame_data.get('damage_taken', 0)
            damage_target_pipes = frame_data.get('damage_target_pipes', 10)
            
            # Calculate continuous progress based on steps/frames
            # First pipe starts at X=280, bird left edge at X=50, bird width = 40
            # Distance for bird to fully clear first pipe: 280 - (50 + 40) = 190 pixels
            # Distance for remaining pipes: (target - 1) * 280 pixels
            
            step = frame_data.get('step', 0)
            game_speed = 5  # pixels per frame
            first_pipe_pos = 280  # first pipe X position
            bird_pos = 50  # bird X position (fixed)
            bird_width = 40  # bird width
            pipe_distance = 280  # pixels between pipes
            
            # Total distance = distance to first pipe + remaining pipes - bird width
            # (bird needs to fully clear, not just have left edge pass)
            total_distance = (first_pipe_pos - bird_pos) + ((damage_target_pipes - 1) * pipe_distance) - bird_width
            
            # Calculate progress as fraction of total distance traveled
            distance_traveled = step * game_speed
            continuous_progress = (distance_traveled / total_distance) * damage_target_pipes
            
            # Cap at target
            continuous_progress = min(continuous_progress, float(damage_target_pipes))
        else:
            # Normal mode, show score and step (unless UI is hidden)
            if not self.hide_ui:
                score = frame_data.get('score', 0)
                draw.text((10, 10), f"Score: {score}", fill=self.colors['text'], font=font)
                draw.text((10, 40), f"Step: {step}", fill=self.colors['text'], font=font)
        
        # Add Q-values if available
        if 'q_do_nothing' in frame_data and 'q_jump' in frame_data:
            q_nothing = frame_data['q_do_nothing']
            q_jump = frame_data['q_jump']
            v_value = frame_data.get('v_value', max(q_nothing, q_jump))
            
            draw.text((10, 70), f"Q(nothing): {q_nothing:.3f}", fill=self.colors['text'], font=font)
            draw.text((10, 100), f"Q(jump): {q_jump:.3f}", fill=self.colors['text'], font=font)
            draw.text((10, 130), f"V-value: {v_value:.3f}", fill=self.colors['text'], font=font)
        
        return img
    
    def render_frame_sequence(self, recording_data, start_step=0, end_step=None):
        """Render a sequence of frames from recording data"""
        episode_data = recording_data.get('episode_data', [])
        
        if end_step is None:
            end_step = len(episode_data)
        
        frames = []
        for i in range(start_step, min(end_step, len(episode_data))):
            frame = self.render_frame(episode_data[i])
            frames.append(frame)
        
        return frames
    
    def create_gif(self, recording_data, output_path, start_step=0, end_step=None, duration=100):
        """Create an animated GIF from recording data"""
        frames = self.render_frame_sequence(recording_data, start_step, end_step)
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved to: {output_path}")
        else:
            print("No frames to save")

# Utility functions for analysis
def analyze_recording(recording_data):
    """Analyze recording data and return statistics"""
    episode_data = recording_data.get('episode_data', [])
    metadata = recording_data.get('metadata', {})
    
    if not episode_data:
        return {'error': 'No episode data found'}
    
    # Basic statistics
    total_steps = len(episode_data)
    final_score = episode_data[-1].get('score', 0) if episode_data else 0
    
    # Action statistics
    actions = [frame.get('action_executed', 0) for frame in episode_data]
    jump_count = sum(1 for action in actions if action == 1)
    jump_rate = jump_count / total_steps if total_steps > 0 else 0
    
    # Score progression
    scores = [frame.get('score', 0) for frame in episode_data]
    score_steps = [i for i, score in enumerate(scores) if i == 0 or score > scores[i-1]]
    
    # Q-value analysis (if available)
    q_analysis = {}
    if any('q_do_nothing' in frame for frame in episode_data):
        q_nothing_values = [frame.get('q_do_nothing', 0) for frame in episode_data if 'q_do_nothing' in frame]
        q_jump_values = [frame.get('q_jump', 0) for frame in episode_data if 'q_jump' in frame]
        v_values = [frame.get('v_value', 0) for frame in episode_data if 'v_value' in frame]
        
        q_analysis = {
            'avg_q_nothing': np.mean(q_nothing_values) if q_nothing_values else 0,
            'avg_q_jump': np.mean(q_jump_values) if q_jump_values else 0,
            'avg_v_value': np.mean(v_values) if v_values else 0,
            'q_nothing_range': [min(q_nothing_values), max(q_nothing_values)] if q_nothing_values else [0, 0],
            'q_jump_range': [min(q_jump_values), max(q_jump_values)] if q_jump_values else [0, 0]
        }
    
    return {
        'metadata': metadata,
        'total_steps': total_steps,
        'final_score': final_score,
        'jump_count': jump_count,
        'jump_rate': jump_rate,
        'score_progression': score_steps,
        'q_analysis': q_analysis
    } 