import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import pygame
import json
import base64
import io
import threading
import queue
import time
from datetime import datetime
import numpy as np
from PIL import Image

# Import original game modules with proper working directory
original_cwd = os.getcwd()
try:
    # Change to parent directory where assets are located
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)
    
    from core import game
    from core import config
    from core import objects
finally:
    # Restore original working directory
    os.chdir(original_cwd)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RecorderGame(game.Game):
    """Extended Game class that records frame data and supports interactive control."""
    
    def __init__(self, agent_name, device, model_path=None, action_fail_probability=0.0, sticky_keys=0, state_size=5, verbose=True, seed=None, stop_at_score=None, np_seed=None, motor_response=None, obs_noise_std=0.0, damage_mode=False, damage_target_pipes=10, bird_sprite_name='bird-flap-1.png'):
        super().__init__(agent_name, device, model_path, action_fail_probability, sticky_keys, state_size, verbose, simple=False, seed=seed, motor_response=motor_response, obs_noise_std=obs_noise_std, damage_mode=damage_mode, damage_target_pipes=damage_target_pipes)
        
        # Store custom bird sprite name
        self.bird_sprite_name = bird_sprite_name
        
        # Ensure the PPO agent is in evaluation mode for consistent inference
        if hasattr(self.agent, 'model'):
            self.agent.model.eval()
        
        # Initialize recording
        self.creation_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.game_id = f"game_{self.creation_timestamp}"
        self.recording_data = {
            'game_id': self.game_id,
            'episode_data': [],
            'metadata': {
                'agent_type': type(self.agent).__name__,
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'action_fail_probability': action_fail_probability,
                'sticky_keys': sticky_keys,
                'model_path': model_path,
                'actual_state_size': getattr(self.agent, 'state_size', state_size),
                'actual_simple': getattr(self.agent, 'simple', False),
                'motor_response': motor_response,
                'damage_mode': damage_mode
            }
        }
        
        # Initialize trajectory recording for follow system
        self.trajectory_data = {
            'metadata': {
                'game_seed': self.seed,
                'agent_type': type(self.agent).__name__,
                'timestamp': datetime.now().isoformat(),
                'recording_start_time': None,
                'recording_end_time': None,
                'total_steps': 0,
                'final_score': 0,
                'success': False,
                'duration_seconds': 0.0
            },
            'trajectory': []
        }
        
        # Queued user input for interactive runs
        self.action_queue = queue.Queue()
        self.current_frame = None
        self.active = False
        self.step_count = 0
        self.trajectory_start_time = None
        
        # Optional early stop threshold (e.g., stop when reaching score N)
        self.stop_at_score = stop_at_score
        # Optional NumPy RNG seed to make limitations (AFP, sticky) deterministic across re-runs
        self.np_seed = np_seed
        
        # Initialize pygame for headless rendering
        pygame.init()
        self.screen = pygame.Surface((config.SCREEN_WIDHT, config.SCREEN_HEIGHT))
        
        # Load images with proper paths - use the same approach as main game
        # Change to parent directory where assets are located
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        
        try:
            # Use custom bird sprite (allows different models to have different bird colors)
            self.bird_image = pygame.image.load(f'assets/{self.bird_sprite_name}')
            self.bird_image = pygame.transform.scale(self.bird_image, (config.BIRD_WIDTH, config.BIRD_HEIGHT))
            self.pipe_image = pygame.image.load('assets/pipe-green.png')
            self.pipe_image = pygame.transform.scale(self.pipe_image, (config.PIPE_WIDHT, config.PIPE_HEIGHT))
            self.ground_image = pygame.image.load('assets/base.png')
            self.ground_image = pygame.transform.scale(self.ground_image, (config.GROUND_WIDHT, config.GROUND_HEIGHT))
            self.background = pygame.image.load('assets/background-day.png')
            self.background = pygame.transform.scale(self.background, (config.SCREEN_WIDHT, config.SCREEN_HEIGHT))
            self.goal_image = pygame.image.load('assets/goal.png')
            self.goal_image = pygame.transform.scale(self.goal_image, (config.PIPE_WIDHT, config.PIPE_GAP))
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def queue_action(self, action):
        """Queue an action from an interactive controller."""
        if not self.action_queue.full():
            self.action_queue.put(action)
    
    def get_current_frame(self):
        """Get current frame as base64 encoded image"""
        if self.current_frame is None:
            return None
        
        return self.current_frame
    
    def is_active(self):
        """Check if game is currently active"""
        return self.active
    
    def capture_frame_data(self, interpolation_factor=0.0, state=None):
        """Capture complete frame data for recording with optional interpolation"""
        self.reward()
        frame_data = {
            'step': self.step_count,
            'frame_index': int(self.step_count * 2 + interpolation_factor * 2),  # Double frame rate
            'timestamp': datetime.now().isoformat(),
            'score': self.score,
            'turn': self.turn,
            'damage_mode': self.damage_mode,
            'damage_target_pipes': self.damage_target_pipes if self.damage_mode else 10,
            
            # Bird data
            'bird': {
                'x': float(vars(self.bird)["pos"][0]),
                'y': float(vars(self.bird)["pos"][1]),
                'width': float(vars(self.bird)["pos"][2]),
                'height': float(vars(self.bird)["pos"][3]),
                'speed': float(vars(self.bird)["speed"])
            },
            
            # Pipes data
            'pipes': [],
            
            # Ground data
            'ground': {
                'x': float(vars(self.ground)["pos"][0]),
                'y': float(vars(self.ground)["pos"][1])
            },
            
            # Game state (for AI analysis) - use provided state or compute if not given
            'state': state if state is not None else self.game_state(),
        }
        
        # Add damage mode specific data
        if self.damage_mode:
            frame_data['pipes_passed'] = self.pipes_passed
            frame_data['damage_taken'] = self.damage_taken
            frame_data['is_in_collision'] = self.is_in_collision
            frame_data['last_damage_step'] = self.last_damage_step
            frame_data['collision_ended_at_frame'] = self.collision_ended_at_frame
            frame_data['goal_x'] = self.goal_x if hasattr(self, 'goal_x') else None
            frame_data['goal_y'] = self.goal_y if hasattr(self, 'goal_y') else None
        
        # Add pipe data
        for pipe in self.pipes:
            pipe_data = {
                'x': float(vars(pipe)["pos"][0]),
                'y': float(vars(pipe)["pos"][1]),
                'width': float(vars(pipe)["pos"][2]),
                'height': float(vars(pipe)["pos"][3]),
                'inverted': pipe.inverted
            }
            frame_data['pipes'].append(pipe_data)
        
        return frame_data
    
    def capture_trajectory_step(self, action, collision=False, pre_update_state=None, pre_update_pipes=None):
        """Capture trajectory data in the format expected by the follow system"""
        if self.trajectory_start_time is None:
            self.trajectory_start_time = time.time()
            self.trajectory_data['metadata']['recording_start_time'] = datetime.now().isoformat()
        
        # Calculate time relative to start
        relative_time = time.time() - self.trajectory_start_time
        
        trajectory_step = {
            'step': self.step_count,
            'time': relative_time,
            'bird': {
                'x': float(vars(self.bird)["pos"][0]),
                'y': float(vars(self.bird)["pos"][1]),
                'speed': float(vars(self.bird)["speed"])
            },
            'action': int(action),
            'game_state': pre_update_state if pre_update_state is not None else self.game_state(),
            'pipes': pre_update_pipes if pre_update_pipes is not None else [],
            'score': self.pipes_passed if self.damage_mode else self.score,
            'collision': collision
        }
        
        # Add damage mode specific data
        if self.damage_mode:
            trajectory_step['damage_taken'] = self.damage_taken
            trajectory_step['pipes_passed'] = self.pipes_passed
        
        # If no pre-update pipes provided, use current pipe positions (legacy behavior)
        if pre_update_pipes is None:
            for pipe in self.pipes:
                pipe_data = {
                    'x': float(vars(pipe)["pos"][0]),
                    'y': float(vars(pipe)["pos"][1]),
                    'width': float(vars(pipe)["pos"][2]),
                    'height': float(vars(pipe)["pos"][3]),
                    'inverted': pipe.inverted
                }
                trajectory_step['pipes'].append(pipe_data)
        
        self.trajectory_data['trajectory'].append(trajectory_step)
        return trajectory_step
    
    def interpolate_frame_data(self, frame1, frame2, factor):
        """Interpolate between two frame data sets"""
        if frame1 is None or frame2 is None:
            return frame1 if frame1 else frame2
        
        interpolated = frame1.copy()
        
        # Interpolate bird position
        bird1 = frame1['bird']
        bird2 = frame2['bird']
        interpolated['bird'] = {
            'x': bird1['x'] + (bird2['x'] - bird1['x']) * factor,
            'y': bird1['y'] + (bird2['y'] - bird1['y']) * factor,
            'width': bird1['width'],
            'height': bird1['height'],
            'speed': bird1['speed'] + (bird2['speed'] - bird1['speed']) * factor
        }
        
        # Interpolate pipe positions
        interpolated['pipes'] = []
        for i, pipe1 in enumerate(frame1['pipes']):
            if i < len(frame2['pipes']):
                pipe2 = frame2['pipes'][i]
                interpolated_pipe = {
                    'x': pipe1['x'] + (pipe2['x'] - pipe1['x']) * factor,
                    'y': pipe1['y'],
                    'width': pipe1['width'],
                    'height': pipe1['height'],
                    'inverted': pipe1['inverted']
                }
                interpolated['pipes'].append(interpolated_pipe)
            else:
                interpolated['pipes'].append(pipe1)
        
        # Interpolate ground position
        ground1 = frame1['ground']
        ground2 = frame2['ground']
        interpolated['ground'] = {
            'x': ground1['x'] + (ground2['x'] - ground1['x']) * factor,
            'y': ground1['y']
        }
        
        # Update frame index for interpolated frame
        interpolated['frame_index'] = int(frame1['step'] * 2 + 1)
        interpolated['step'] = frame1['step']  # Keep same step number
        interpolated['score'] = frame1['score']  # Keep same score
        interpolated['turn'] = frame1['turn']  # Keep same turn
        interpolated['state'] = frame1['state']  # Keep same state
        interpolated['action_intended'] = frame1.get('action_intended', 0)
        interpolated['action_executed'] = frame1.get('action_executed', 0)
        interpolated['action_failed'] = frame1.get('action_failed', False)
        
        # Keep damage mode data from original frame
        interpolated['damage_mode'] = frame1.get('damage_mode', False)
        interpolated['damage_target_pipes'] = frame1.get('damage_target_pipes', 10)
        if interpolated['damage_mode']:
            interpolated['pipes_passed'] = frame1.get('pipes_passed', 0)
            interpolated['damage_taken'] = frame1.get('damage_taken', 0)
            interpolated['is_in_collision'] = frame1.get('is_in_collision', False)
            interpolated['last_damage_step'] = frame1.get('last_damage_step', -999)
            interpolated['collision_ended_at_frame'] = frame1.get('collision_ended_at_frame', -999)
            # Interpolate goal position if present
            goal_x_1 = frame1.get('goal_x', None)
            goal_x_2 = frame2.get('goal_x', None)
            if goal_x_1 is not None and goal_x_2 is not None:
                interpolated['goal_x'] = goal_x_1 + (goal_x_2 - goal_x_1) * factor
            else:
                interpolated['goal_x'] = goal_x_1
            interpolated['goal_y'] = frame1.get('goal_y', None)
        
        # Keep Q-values from original frame
        if 'q_do_nothing' in frame1:
            interpolated['q_do_nothing'] = frame1['q_do_nothing']
            interpolated['q_jump'] = frame1['q_jump']
            interpolated['v_value'] = frame1['v_value']
        
        return interpolated
    
    def render_current_frame(self):
        """Render current game state to a frame"""
        # Clear screen with background
        self.screen.blit(self.background, (0, 0))

        # Draw game objects
        # In damage mode, draw bird on top of pipes for visibility during collisions
        if self.damage_mode:
            self.ground.draw(self.screen)
            for pipe in self.pipes:
                pipe.draw(self.screen)
            # Draw goal finish line at the final pipes (if visible)
            if hasattr(self, 'goal_x') and self.goal_x is not None and -config.PIPE_WIDHT < self.goal_x < config.SCREEN_WIDHT:
                self.screen.blit(self.goal_image, (self.goal_x, self.goal_y))
            self.bird.draw(self.screen)  # Bird on top
        else:
            self.bird.draw(self.screen)
            self.ground.draw(self.screen)
            for pipe in self.pipes:
                pipe.draw(self.screen)

        # Draw score (or progress bar/damage in damage mode)
        font = pygame.font.Font(None, 36)
        if self.damage_mode:
            # In damage mode, show progress bar and damage icons below
            
            # Calculate continuous progress based on steps/frames
            # First pipe starts at X=280, bird left edge at X=50, bird width = 40
            # Distance for bird to fully clear first pipe: 280 - (50 + 40) = 190 pixels
            # Distance for remaining pipes: (target - 1) * 280 pixels
            
            game_speed = config.GAME_SPEED  # 5 pixels per frame
            first_pipe_pos = config.PIPE_DISTANCE  # 280 (first pipe X position)
            bird_pos = 50  # bird X position (fixed)
            bird_width = config.BIRD_WIDTH  # 40 pixels
            pipe_distance = config.PIPE_DISTANCE  # 280 pixels between pipes
            
            # Total distance = distance to first pipe + remaining pipes - bird width
            # (bird needs to fully clear, not just have left edge pass)
            total_distance = (first_pipe_pos - bird_pos) + ((self.damage_target_pipes - 1) * pipe_distance) - bird_width
            
            # Calculate progress as fraction of total distance traveled
            distance_traveled = self.turn * game_speed
            continuous_progress = (distance_traveled / total_distance) * self.damage_target_pipes
            
            # Cap at target
            continuous_progress = min(continuous_progress, float(self.damage_target_pipes))
        else:
            # Normal mode, show score
            score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
        # Convert pygame surface to base64 image
        img_str = pygame.image.tostring(self.screen, 'RGB')
        img = Image.frombytes('RGB', (config.SCREEN_WIDHT, config.SCREEN_HEIGHT), img_str)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        self.current_frame = f'data:image/png;base64,{img_base64}'
    
    def main(self, draw=False, draw_value=False, save_values=True, record_frames=True, max_score=100):
        """Main game loop with recording capabilities"""
        
        self.active = True
        self.init_game()
        # Set NumPy RNG for deterministic limitations behavior if requested
        if self.np_seed is not None:
            np.random.seed(self.np_seed)
            # Also seed the dedicated limitations RNGs for deterministic motor response and observation noise
            self.set_limitations_seed(self.np_seed)
        
        # Initialize recording
        if save_values:
            self.recording_data['episode_data'] = []
            self.step_count = 0
        
        active_episode = True
        episode_ended_by_collision = False  # Track if episode ended due to collision
        
        # Track last action for final frame capture (if needed)
        last_original_action = 0
        last_executed_action = 0
        last_action_failed = False
        last_action_blocked = False
        
        # Game loop
        while active_episode and self.active:
            
            # Get agent action and capture pre-update state and object positions
            state = self.game_state()
            
            # Capture pre-update bird state
            pre_bird_x = float(vars(self.bird)["pos"][0])
            pre_bird_y = float(vars(self.bird)["pos"][1])
            pre_bird_speed = float(vars(self.bird)["speed"])
            
            # Capture pipe positions BEFORE environment update (aligned with state timing)
            pre_update_pipes = []
            for pipe in self.pipes:
                pipe_data = {
                    'x': float(vars(pipe)["pos"][0]),
                    'y': float(vars(pipe)["pos"][1]),
                    'width': float(vars(pipe)["pos"][2]),
                    'height': float(vars(pipe)["pos"][3]),
                    'inverted': pipe.inverted
                }
                pre_update_pipes.append(pipe_data)
            
            # Capture pre-update collision state (for damage mode)
            if self.damage_mode:
                pre_is_in_collision = self.is_in_collision
                pre_last_damage_step = self.last_damage_step
                pre_collision_ended_at_frame = self.collision_ended_at_frame
                pre_damage_taken = self.damage_taken
                pre_pipes_passed = self.pipes_passed
                pre_goal_x = self.goal_x
                pre_goal_y = self.goal_y
            
            action = self.agent.act(state, self.train)
            
            # Determine which action is actually executed this step
            original_action = action
            executed_action = 0
            action_blocked_by_cooldown = False
            action_failed = False

            if self.is_ppo and self.motor_response is not None:
                # Motor-response limitation: delayed execution for jump
                if self.ppo_jump_pending:
                    # Ignore any new jump selections while pending
                    self.ppo_pending_delay_remaining -= 1
                    if self.ppo_pending_delay_remaining <= 0:
                        executed_action = 1
                        self.ppo_jump_pending = False
                        self.ppo_pending_delay_remaining = 0
                else:
                    if action == 1:
                        mean_delay, std_delay = self.motor_response
                        delay = int(max(0, round(self._rng_motor.normal(mean_delay, std_delay))))
                        if delay <= 0:
                            executed_action = 1
                        else:
                            self.ppo_jump_pending = True
                            self.ppo_pending_delay_remaining = delay
                            executed_action = 0
                    elif action == -1:
                        executed_action = -1
            else:
                executed_action = action

            # Apply sticky keys and action failure to the action that would be executed this step
            if executed_action == 1 and self.sticky_keys > 0 and self.jump_cooldown_remaining > 0:
                executed_action = 0
                action_blocked_by_cooldown = True
            if executed_action == 1 and self.action_fail_probability > 0:
                if self._rng_action_fail.random() < self.action_fail_probability:
                    executed_action = 0
                    action_failed = True
            
            # Execute action
            if executed_action == 1:
                self.bird.bump()
                # Set cooldown period after successful jump (random between sticky_keys-2 and sticky_keys+2)
                # Only set cooldown if the action actually succeeded (not blocked by cooldown or failed)
                # Use separate random number generator to avoid affecting pipe generation
                if self.sticky_keys > 0 and not action_blocked_by_cooldown and not action_failed:
                    min_cooldown = max(1, self.sticky_keys - 2)  # Ensure at least 1 step cooldown
                    max_cooldown = self.sticky_keys + 2
                    self.jump_cooldown_remaining = self._rng_sticky.integers(min_cooldown, max_cooldown + 1)
            if executed_action == -1:
                active_episode = False
                break
            
            # Update sticky keys cooldown
            if self.sticky_keys > 0 and self.jump_cooldown_remaining > 0:
                self.jump_cooldown_remaining -= 1
            
            # Update environment
            self.bird.update()
            for pipe in self.pipes:
                pipe.update()
            self.score_update()
            
            # Update goal position in damage mode (move it with the pipes)
            if self.damage_mode and self.goal_x is not None:
                self.goal_x -= config.GAME_SPEED
            
            # Check for pipe damage in damage mode (must be before pipe_handling)
            if self.damage_mode:
                self.check_pipe_damage()
            
            self.pipe_handling()
            
            # Check for collisions
            collision_occurred = self.collision()
            
            # Capture frame data if recording (always capture before checking collision end)
            if record_frames:
                # Capture the main frame with pre-update state to avoid extra RNG consumption
                frame_data = self.capture_frame_data(state=state)
                # Override bird and pipes with pre-update values for perfect alignment with trajectory
                frame_data['bird'] = {
                    'x': pre_bird_x,
                    'y': pre_bird_y,
                    'speed': pre_bird_speed,
                }
                frame_data['pipes'] = pre_update_pipes
                
                # Override collision state with pre-update values (for damage mode)
                if self.damage_mode:
                    frame_data['is_in_collision'] = pre_is_in_collision
                    frame_data['last_damage_step'] = pre_last_damage_step
                    frame_data['collision_ended_at_frame'] = pre_collision_ended_at_frame
                    frame_data['damage_taken'] = pre_damage_taken
                    frame_data['pipes_passed'] = pre_pipes_passed
                    frame_data['goal_x'] = pre_goal_x
                    frame_data['goal_y'] = pre_goal_y
                
                # Add action data
                frame_data['action_intended'] = int(original_action)
                frame_data['action_executed'] = int(executed_action)
                frame_data['action_failed'] = action_failed
                frame_data['action_blocked_by_cooldown'] = action_blocked_by_cooldown
                frame_data['pending_jump'] = self.ppo_jump_pending if self.is_ppo else False
                frame_data['pending_delay_remaining'] = self.ppo_pending_delay_remaining if self.is_ppo else 0
                frame_data['jump_cooldown_remaining'] = self.jump_cooldown_remaining
                
                self.recording_data['episode_data'].append(frame_data)
                
                # If collision occurred, also capture a post-update collision frame so videos can show the hit
                if collision_occurred and record_frames:
                    # Get post-collision state and capture frame without extra RNG consumption
                    collision_state = self.game_state()
                    collision_frame = self.capture_frame_data(state=collision_state)
                    collision_frame['collision'] = True
                    # Keep action metadata consistent
                    collision_frame['action_intended'] = int(original_action)
                    collision_frame['action_executed'] = int(executed_action)
                    collision_frame['action_failed'] = action_failed
                    collision_frame['action_blocked_by_cooldown'] = action_blocked_by_cooldown
                    collision_frame['pending_jump'] = self.ppo_jump_pending if self.is_ppo else False
                    collision_frame['pending_delay_remaining'] = self.ppo_pending_delay_remaining if self.is_ppo else 0
                    collision_frame['jump_cooldown_remaining'] = self.jump_cooldown_remaining
                    self.recording_data['episode_data'].append(collision_frame)
                
                # Capture trajectory data for follow system (with collision status and pre-update data)
                self.capture_trajectory_step(executed_action, collision=collision_occurred, 
                                           pre_update_state=state, pre_update_pipes=pre_update_pipes)
                
                self.step_count += 1
            
            # End episode if collision occurred
            if collision_occurred:
                active_episode = False
                episode_ended_by_collision = True
            
            # Render current frame for interactive viewing only if needed
            if draw or record_frames:
                self.render_current_frame()
            
            # Track last action for potential final frame
            last_original_action = original_action
            last_executed_action = executed_action
            last_action_failed = action_failed
            last_action_blocked = action_blocked_by_cooldown
            
            self.turn += 1
            
            # Terminate episode after reaching a threshold
            if self.damage_mode:
                # In damage mode, end after passing 10 pipes
                if self.pipes_passed >= self.damage_target_pipes:
                    active_episode = False
            else:
                # Normal mode: check score threshold
                threshold = self.stop_at_score if self.stop_at_score is not None else max_score
                if self.score >= threshold:
                    active_episode = False
            
            # Small delay only when drawing/recording; skip in fast evals
            if draw or record_frames:
                time.sleep(0.033)  # ~30 FPS for smooth gameplay
        
        # If game ended by reaching target pipes (not collision), capture final frame with updated pipes_passed
        if record_frames and self.damage_mode and not episode_ended_by_collision:
            if self.pipes_passed >= self.damage_target_pipes:
                # Capture one final frame showing the updated pipes_passed count
                final_state = self.game_state()
                final_frame = self.capture_frame_data(state=final_state)
                # Keep the last action metadata
                final_frame['action_intended'] = int(last_original_action)
                final_frame['action_executed'] = int(last_executed_action)
                final_frame['action_failed'] = last_action_failed
                final_frame['action_blocked_by_cooldown'] = last_action_blocked
                final_frame['pending_jump'] = self.ppo_jump_pending if self.is_ppo else False
                final_frame['pending_delay_remaining'] = self.ppo_pending_delay_remaining if self.is_ppo else 0
                final_frame['jump_cooldown_remaining'] = self.jump_cooldown_remaining
                self.recording_data['episode_data'].append(final_frame)
                self.step_count += 1
        
        self.active = False
        
        # Save recording data
        if save_values:
            self.save_recording()
        
        # Return appropriate score based on mode
        if self.damage_mode:
            return self.damage_taken  # In damage mode, return damage as "score"
        else:
            return self.score
    
    def save_recording(self):
        """Save the complete game recording"""
        # Add final metadata
        if self.damage_mode:
            # In damage mode, final_score is pipes passed, and we track damage separately
            self.recording_data['metadata']['final_score'] = self.pipes_passed
            self.recording_data['metadata']['pipes_passed'] = self.pipes_passed
            self.recording_data['metadata']['damage_taken'] = self.damage_taken
        else:
            self.recording_data['metadata']['final_score'] = self.score
        
        self.recording_data['metadata']['total_steps'] = self.step_count
        self.recording_data['metadata']['episode_completed'] = True
        
        # Store the trajectory filename for easy linking
        agent_type = type(self.agent).__name__.lower()
        trajectory_filename = f"recording_{agent_type}_score{self.score}_seed{self.seed}_{self.creation_timestamp}.json"
        self.recording_data['metadata']['trajectory_filename'] = trajectory_filename
        
        # Use recordings directory inside the recording package
        recording_dir = os.path.dirname(os.path.abspath(__file__))
        recordings_dir = os.path.join(recording_dir, 'recordings')
        
        # Save frame recording to file
        filename = os.path.join(recordings_dir, f'{self.game_id}.json')
        with open(filename, 'w') as f:
            json.dump(self.recording_data, f, indent=2, cls=NumpyEncoder)
        
        # Finalize and save trajectory data for follow system
        self.save_trajectory()
    
    def save_trajectory(self):
        """Save trajectory data in the format expected by the follow system"""
        if self.trajectory_start_time is None:
            print("No trajectory data recorded")
            return
            
        # Finalize trajectory metadata
        end_time = time.time()
        duration = end_time - self.trajectory_start_time
        
        self.trajectory_data['metadata']['recording_end_time'] = datetime.now().isoformat()
        self.trajectory_data['metadata']['total_steps'] = len(self.trajectory_data['trajectory'])  # Actual trajectory length
        
        # Handle final score based on mode
        if self.damage_mode:
            self.trajectory_data['metadata']['final_score'] = self.pipes_passed  # Pipes passed in damage mode
            self.trajectory_data['metadata']['damage_taken'] = self.damage_taken
            self.trajectory_data['metadata']['success'] = self.pipes_passed >= self.damage_target_pipes and self.damage_taken == 0  # Perfect run
        else:
            self.trajectory_data['metadata']['final_score'] = self.score
            self.trajectory_data['metadata']['success'] = self.score >= 100  # Consider success if score >= 100
        
        self.trajectory_data['metadata']['duration_seconds'] = duration
        
        # Create trajectory filename using the same timestamp as the recording
        agent_type = type(self.agent).__name__.lower()
        trajectory_filename = f"recording_{agent_type}_score{self.score}_seed{self.seed}_{self.creation_timestamp}.json"
        
        # Save to follow/trajectories directory
        recording_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(recording_dir)
        trajectories_dir = os.path.join(project_root, 'follow', 'trajectories')
        
        # Create trajectories directory if it doesn't exist
        os.makedirs(trajectories_dir, exist_ok=True)
        
        trajectory_path = os.path.join(trajectories_dir, trajectory_filename)
        
        # Save trajectory file
        with open(trajectory_path, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2, cls=NumpyEncoder)
    
        return trajectory_path

# Helper function to create web user agent
class WebUserAgent:
    """User agent that receives externally queued actions."""
    
    def __init__(self):
        pass
    
    def act(self, state, train):
        # Actions will be queued from an external controller
        return 0 