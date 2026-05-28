from core.config import *
from core import objects
import pygame

# Suppress pygame hello message
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from pygame.locals import *
import random
import sys
import time
import numpy as np

# Suppress libpng iCCP warnings
class _FilterStderr:
    def write(self, s):
        if 'libpng warning: iCCP' not in s:
            sys.__stderr__.write(s)
    def flush(self): sys.__stderr__.flush()
sys.stderr = _FilterStderr()
import torch
import json
from datetime import datetime
from agents import ppo_agent
import copy

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')


#Agents
AGENTS = ["ppo_agent"]

#Pygame image loading
bird_image = pygame.image.load(os.path.join(ASSET_DIR, 'bird-flap-1.png'))
bird_image = pygame.transform.scale(bird_image, (BIRD_WIDTH, BIRD_HEIGHT))
pipe_image = pygame.image.load(os.path.join(ASSET_DIR, 'pipe-green.png'))
pipe_image = pygame.transform.scale(pipe_image, (PIPE_WIDHT, PIPE_HEIGHT))
ground_image = pygame.image.load(os.path.join(ASSET_DIR, 'base.png'))
ground_image = pygame.transform.scale(ground_image, (GROUND_WIDHT, GROUND_HEIGHT))
BACKGROUND = pygame.image.load(os.path.join(ASSET_DIR, 'background-day.png'))
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
goal_image = pygame.image.load(os.path.join(ASSET_DIR, 'goal.png'))
goal_image = pygame.transform.scale(goal_image, (PIPE_WIDHT, PIPE_GAP))



"""
Main game class which is running and controlling the game
"""
class Game:
    
    def __init__(self, agent_name, device, model_path=None, action_fail_probability=0.0, sticky_keys=0, state_size=5, verbose=True, simple=False, seed=None, motor_response=None, obs_noise_std=0.0, obs_noise_horizontal_ratio=0.0, damage_mode=False, damage_target_pipes=10):

        #Initialize agent
        if not agent_name in AGENTS: sys.exit("Agent not defined")
        if device != "cpu" and device != "cuda" and device != "mps": sys.exit("Computing device not available")
        if agent_name == "ppo_agent":
            # PPO observes an extra pending flag feature
            self.agent = ppo_agent.PPO_agent(device=device, state_size=state_size + 1, verbose=verbose)
            if verbose:
                print("Initialize game with: PPO_agent")
            # Load PPO weights if provided
            if model_path and os.path.exists(model_path):
                try:
                    self.agent.load_model(model_path)
                    if verbose:
                        print(f"Loaded PPO model weights from: {model_path}")
                except Exception as e:
                    if verbose:
                        print(f"Failed to load PPO model: {e}")
        self.device = device
        self.is_ppo = isinstance(self.agent, ppo_agent.PPO_agent)
        
        # State size configuration (3 or 5 features)
        if state_size not in [3, 5]:
            sys.exit("State size must be 3 or 5")
        self.state_size = state_size
        
        # Action failure probability for simulating imperfect human play
        self.action_fail_probability = action_fail_probability
            
        # Sticky keys parameter - prevents jumping for N±2 steps after each jump
        self.sticky_keys = sticky_keys

        # Motor response limitation (only applied for PPO agent)
        # motor_response is expected as tuple: (mean_delay, std_delay) in time steps
        self.motor_response = None
        if self.is_ppo and motor_response is not None:
            try:
                mean_delay, std_delay = motor_response
                self.motor_response = (float(mean_delay), float(std_delay))
            except Exception:
                if verbose:
                    print("Invalid motor_response; expected tuple(mean, std). Disabling.")
                self.motor_response = None
        # RNG for motor response sampling (avoid interfering with game RNG)
        self._rng_motor = np.random.default_rng()

        # Observation noise parameter for simulating perception limitations
        self.obs_noise_std = float(obs_noise_std)
        # Ratio for horizontal noise relative to vertical noise
        # horizontal_noise_std = obs_noise_std * obs_noise_horizontal_ratio
        self.obs_noise_horizontal_ratio = float(obs_noise_horizontal_ratio)
        # RNG for observation noise (avoid interfering with game RNG)
        # Use unseeded RNG so each run has different noise, but doesn't affect map generation
        self._rng_obs_noise = np.random.default_rng()
        
        # RNG for action failure probability (avoid interfering with game RNG)
        self._rng_action_fail = np.random.default_rng()
        
        # RNG for sticky keys cooldown (avoid interfering with game RNG)
        self._rng_sticky = np.random.default_rng()

        # Damage mode configuration
        self.damage_mode = damage_mode
        self.damage_target_pipes = damage_target_pipes
        if damage_mode and verbose:
            print(f"Game initialized in DAMAGE MODE: Agent survives pipe collisions but takes damage (target: {damage_target_pipes} pipes)")

        #Game objects (Get initialized new every game played)
        self.bird = None
        self.ground = None
        self.pipes = None
        self.score = None
        self.turn = None
        self.verbose = verbose

        #Training mode for agent
        self.train = False
        
        # Set random seed for reproducible level generation
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def init_game(self):

        # Reset random seed for reproducible level generation if seed is set
        if self.seed is not None:
            random.seed(self.seed)

        #Initialize game objects
        self.bird = objects.Bird(bird_image)
        self.ground = objects.Ground(ground_image, 0)
        self.pipes = []
        self.score = 0
        self.turn = 0
        
        # Initialize sticky keys tracking
        self.jump_cooldown_remaining = 0  # Steps remaining before next jump is allowed

        # Initialize PPO motor-response pending-jump tracking
        self.ppo_jump_pending = False
        self.ppo_pending_delay_remaining = 0

        # Initialize damage mode tracking
        if self.damage_mode:
            self.damage_taken = 0  # Track total damage
            self.pipes_passed = 0  # Track pipes successfully passed
            self.last_damage_step = -999  # Track when damage last occurred (for cooldown)
            self.is_in_collision = False  # Track if currently colliding with pipe
            self.collision_ended_at_frame = -999  # Track when collision ended (for sprite persistence)
            self.pipes_created_count = 0  # Track total pipes created to identify the final one
            # Calculate goal position upfront based on where final pipe will be
            # Pipes are positioned at PIPE_DISTANCE * N, so Nth pipe is at N * PIPE_DISTANCE
            self.goal_x = PIPE_DISTANCE * self.damage_target_pipes  # X position of the goal
            # goal_y will be updated when the actual pipe is created (depends on random ysize)
            # For now, set it to middle of screen as a reasonable default
            self.goal_y = (SCREEN_HEIGHT - GROUND_HEIGHT) // 2  # Y position of the goal (top of gap)

        #Initialize pipes
        for i in range(3):
            #Pipe initial positions
            xpos = PIPE_DISTANCE * i + PIPE_DISTANCE
            ysize = random.randint(200, 300)

            #Append pipes to list
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))
            
            # Track pipe creation in damage mode
            if self.damage_mode:
                self.pipes_created_count += 1
                # If this is the final pipe, update goal_y to match actual pipe gap position
                if self.pipes_created_count == self.damage_target_pipes:
                    # goal_x is already set correctly at init, just update goal_y
                    self.goal_y = ysize  # Top of the gap (matches top pipe's bottom edge)

        # In training mode, randomize initial bird position across the whole vertical axis
        # (clamped to valid in-game bounds) and initial vertical speed, without perturbing other RNGs
        if self.train:
            rng = np.random.default_rng()

            # Sample anywhere along the valid vertical range (avoid roof/ground by a small margin)
            margin = 2
            min_y = margin
            max_y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT - margin
            if max_y < min_y:
                # Fallback safety in extreme configurations
                min_y = 0
                max_y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT
            start_y = int(rng.integers(min_y, max_y + 1))
            vars(self.bird)["pos"][1] = start_y

            # Randomize initial vertical speed uniformly in [-SPEED, SPEED] (jump speed range)
            vars(self.bird)["speed"] = float(rng.uniform(-SPEED, SPEED))

    def pipe_handling(self):
        #if pipes out of screen add new ones and remove old
        if vars(self.pipes[0])["pos"][0] <= -100:

            #Remove old pipes
            del self.pipes[0]
            del self.pipes[0]

            #New pipe initial positions
            xpos = PIPE_DISTANCE * 3 - 100
            ysize = random.randint(150, 350)

            #Append new pipes
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))
            
            # Track pipe creation in damage mode and update goal position
            if self.damage_mode:
                self.pipes_created_count += 1
                # If this is the final pipe, update goal_y to match actual pipe gap position
                if self.pipes_created_count == self.damage_target_pipes:
                    # goal_x is already set correctly at init, just update goal_y
                    self.goal_y = SCREEN_HEIGHT - ysize - PIPE_GAP  # Top of the gap (matches pipe position)
                
    def collision(self):
        #Check ground and roof collision (always fatal)
        if vars(self.bird)["pos"][1] < 0 or vars(self.bird)["pos"][1] > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT: # 0 is the ceiling of the screen, y axis is going from top to bottom
            return True

        #Check for pipe collision
        if vars(self.pipes[0])["pos"][0] - vars(self.bird)["pos"][2] < vars(self.bird)["pos"][0] and vars(self.bird)["pos"][0] < vars(self.pipes[0])["pos"][0] +vars(self.pipes[0])["pos"][2]:
            if vars(self.pipes[0])["pos"][1] < vars(self.bird)["pos"][1] + vars(self.bird)["pos"][3] or vars(self.pipes[0])["pos"][1] - PIPE_GAP > vars(self.bird)["pos"][1]:
                # In damage mode, pipe collision doesn't end the game
                if not self.damage_mode:
                    return True

        return False

    def check_pipe_damage(self):
        """Check if bird is colliding with a pipe and apply damage (damage mode only)"""
        if not self.damage_mode:
            return False
        
        bird_x = vars(self.bird)["pos"][0]
        bird_y = vars(self.bird)["pos"][1]
        bird_width = vars(self.bird)["pos"][2]
        bird_height = vars(self.bird)["pos"][3]
        
        # Check all pipe pairs (pipes come in pairs of 2: top and bottom at same X)
        # Only check non-inverted pipes (top pipes) since they have the reference Y coordinate
        collision_detected = False
        
        for pipe in self.pipes:
            # Skip inverted (bottom) pipes - we only need to check the top pipe of each pair
            if pipe.inverted:
                continue
                
            pipe_x = vars(pipe)["pos"][0]
            pipe_y = vars(pipe)["pos"][1]
            pipe_width = vars(pipe)["pos"][2]
            
            # Check horizontal overlap (same as original collision() method)
            if pipe_x - bird_width < bird_x and bird_x < pipe_x + pipe_width:
                # Check vertical collision (EXACT same logic as original collision() method)
                # pipe_y is the bottom edge of the top pipe
                # PIPE_GAP is the size of the gap between top and bottom pipes
                # Gap is from (pipe_y - PIPE_GAP) to pipe_y
                # Collision occurs if:
                # 1. Bird's bottom extends above the gap (hits top pipe): pipe_y < bird_y + bird_height
                # 2. Bird's top is below the gap start (hits bottom pipe): pipe_y - PIPE_GAP > bird_y
                # The bird is SAFE in the gap if: pipe_y - PIPE_GAP <= bird_y and bird_y + bird_height <= pipe_y
                
                if pipe_y < bird_y + bird_height or pipe_y - PIPE_GAP > bird_y:
                    collision_detected = True
                    break
        
        # Track collision state transitions for sprite persistence
        was_in_collision = self.is_in_collision
        self.is_in_collision = collision_detected
        
        # When collision ends, record the frame for sprite persistence (keep showing collision sprite for 3 more frames)
        if was_in_collision and not collision_detected:
            self.collision_ended_at_frame = self.turn
        
        # Apply damage only if collision detected and cooldown expired
        if collision_detected:
            # Damage cooldown: prevent multiple damages in quick succession
            # Compute cooldown based on collision duration:
            # - Collision lasts roughly (PIPE_WIDTH + BIRD_WIDTH) / GAME_SPEED frames
            # - Add 50% safety margin to ensure only 1 damage per pipe
            collision_duration = (PIPE_WIDHT + BIRD_WIDTH) / GAME_SPEED
            DAMAGE_COOLDOWN = int(collision_duration * 1.5)
            
            if self.turn - self.last_damage_step >= DAMAGE_COOLDOWN:
                # Apply damage
                self.damage_taken += 1
                self.last_damage_step = self.turn
                if self.verbose:
                    print(f"💥 Damage! Hit pipe (step={self.turn}). Total damage: {self.damage_taken}")
                return True
        
        return False

    def score_update(self):
        if vars(self.bird)["pos"][0] == vars(self.pipes[0])["pos"][0]:
            if self.damage_mode:
                self.pipes_passed += 1
            else:
                self.score += 1

    def game_state(self):
        state = []

        #Gamestate passing to the agent: 1-horizontal distance to next pipe, 2-vertical distance to lower next pipe, 3-bird speed, 4-distance to ground, 5-distance to ceiling
        for pipe in self.pipes:
            if vars(self.bird)["pos"][0] - BIRD_WIDTH < vars(pipe)["pos"][0]: #Check which pipe is the next one
                # [LEGACY]
                # state.append((- vars(self.bird)["pos"][0] + vars(pipe)["pos"][2] + vars(pipe)["pos"][0]) / PIPE_DISTANCE)
                # [NEW]
                horizontal_distance = vars(pipe)["pos"][0] - vars(self.bird)["pos"][0] - BIRD_WIDTH
                # print('horizontal distance:', horizontal_distance)
                
                # Apply observation noise to horizontal distance if enabled
                if self.obs_noise_std > 0.0:
                    # Add noise to the raw horizontal distance (before normalization)
                    # Horizontal noise std = obs_noise_std * obs_noise_horizontal_ratio
                    horizontal_noise_std = self.obs_noise_std * self.obs_noise_horizontal_ratio
                    horizontal_noise = self._rng_obs_noise.normal(0.0, horizontal_noise_std)
                    # print('horizontal noise:', horizontal_noise)
                    horizontal_distance = horizontal_distance + horizontal_noise
                horizontal_distance = horizontal_distance / PIPE_DISTANCE
                
                state.append(horizontal_distance)
                
                ### Calculate vertical distance to lower next pipe
                # [LEGACY]
                # pipe_height_distance = (vars(pipe)["pos"][1] - PIPE_GAP/2 - vars(self.bird)["pos"][1] - vars(self.bird)["pos"][3] / 2) / SCREEN_HEIGHT * 2
                
                # [NEW]
                pipe_height_distance = vars(pipe)["pos"][1] - vars(self.bird)["pos"][1] - BIRD_HEIGHT
                # print('pipe height distance:', pipe_height_distance)
                # 
                # Apply observation noise to pipe height perception if enabled
                if self.obs_noise_std > 0.0:
                    # Add noise to the raw pipe height distance (before normalization)
                    vertical_noise = self._rng_obs_noise.normal(0.0, self.obs_noise_std)
                    pipe_height_distance = pipe_height_distance + vertical_noise
                pipe_height_distance = pipe_height_distance / SCREEN_HEIGHT
                
                state.append(pipe_height_distance)
                break
        state.append(vars(self.bird)["speed"] / SPEED)
        
        # Add distance to ground and ceiling only if state_size is 5
        if self.state_size == 5:
            distance_to_ground = (SCREEN_HEIGHT - GROUND_HEIGHT - vars(self.bird)["pos"][1]) - BIRD_HEIGHT
            state.append(distance_to_ground / SCREEN_HEIGHT)
            state.append(vars(self.bird)["pos"][1] / SCREEN_HEIGHT)

        # Append pending flag used by the PPO motor-response delay model.
        if self.is_ppo:
            state.append(1.0 if self.ppo_jump_pending else 0.0)

        return state

    def reward(self, action=0):

        reward = 0.1 #reward of 0.05 for surviving

        #########################################################
        ### SPEED REWARD
        #########################################################
        
        # reward_speed = 0.2 * (max((abs(vars(self.bird)["speed"]) - 15), 0)/15)**2
        # if reward_speed > 0:
        #     print('reward speed:', reward_speed)
        # reward += reward_speed
        
        #########################################################
        ### REWARD FOR GETTING CLOSE TO THE BOTTOM PIPE
        #########################################################

        # horizontal_distance = vars(self.pipes[0])["pos"][0] - vars(self.bird)["pos"][0] - BIRD_WIDTH
        # if horizontal_distance <= - BIRD_WIDTH / 2 and horizontal_distance >= - PIPE_WIDHT - BIRD_WIDTH / 2:
        #     vertical_distance = vars(self.pipes[0])["pos"][1] - vars(self.bird)["pos"][1] - BIRD_HEIGHT
        #     # reward_bottom_pipe = np.exp(-vertical_distance / BIRD_HEIGHT) / 10
        #     reward_bottom_pipe = - vertical_distance / (BIRD_HEIGHT * 5)
        #     reward_bottom_pipe = 0.1 * (vars(self.bird)["pos"][1] - (vars(self.pipes[0])["pos"][1] - PIPE_GAP)) / PIPE_GAP

        #     reward += reward_bottom_pipe
        
        # reward_ground = 0.1 * vars(self.bird)["pos"][1] / SCREEN_HEIGHT
        # reward += reward_ground

        #########################################################
        ### REWARD FOR GETTING CLOSE TO THE TOP PIPE
        #########################################################

        # reward = 0.0
        # horizontal_distance = vars(self.pipes[0])["pos"][0] - vars(self.bird)["pos"][0] - BIRD_WIDTH
        # print('horizontal distance:', horizontal_distance)
        # if horizontal_distance <= - BIRD_WIDTH / 4 and horizontal_distance >= - PIPE_WIDHT - BIRD_WIDTH / 4 :
            # vertical_distance = vars(self.bird)["pos"][1] - vars(self.pipes[0])["pos"][1] + PIPE_GAP
            # reward_top_pipe = 1.0 * ((PIPE_GAP - vertical_distance) / PIPE_GAP) ** 4

            # print('horizontal distance:', horizontal_distance)
            # print('vertical distance:', vertical_distance)
            # print('reward top pipe:', ((PIPE_GAP - vertical_distance) / PIPE_GAP) ** 4)
            # print()
            

            # print('bird y:', vars(self.bird)["pos"][1])
            # print('pipe y:', vars(self.pipes[0])["pos"][1])
            # print('vertical distance:', vars(self.bird)["pos"][1] - vars(self.pipes[0])["pos"][1] + PIPE_GAP)
            # print('reward top pipe:', reward_top_pipe)
            # print('--------------------------------')
            # reward += reward_top_pipe

        # Small reward for jumping
        # if action == 1:
        #     reward += 0.2

        # Limit the speed
        # reward_speed = 0.2 * (max((abs(vars(self.bird)["speed"]) - 15), 0)/12)**2
        # reward -= reward_speed

        # distance_to_top = vars(self.bird)["pos"][1]
        # reward_top = 0.1 * (SCREEN_HEIGHT - distance_to_top) / SCREEN_HEIGHT
        # reward_top = 0.1 * ((SCREEN_HEIGHT - distance_to_top) / SCREEN_HEIGHT) ** 4
        # print('reward top:', (SCREEN_HEIGHT - distance_to_top) / SCREEN_HEIGHT)
        # reward += reward_top

        # if self.collision():
        #     reward = -1 #reward -10 for colliding

        return round(reward,4)

    def set_seed(self, seed):
        """Set or change the random seed for reproducible level generation"""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            if self.verbose:
                print(f"Game seed changed to: {seed}")
        else:
            if self.verbose:
                print("Game seed disabled (using system random)")
        # Note: observation noise RNG is intentionally NOT updated - it should remain random

    def set_limitations_seed(self, limitations_seed):
        """Set seed for limitations RNGs (motor response, observation noise, action fail, sticky keys) for deterministic replays"""
        if limitations_seed is not None:
            # Seed motor response RNG
            self._rng_motor = np.random.default_rng(limitations_seed)
            # Seed observation noise RNG (even though we want it random between runs, 
            # we need it deterministic between quick/full runs for the same attempt)
            self._rng_obs_noise = np.random.default_rng(limitations_seed + 1)
            # Seed action failure RNG
            self._rng_action_fail = np.random.default_rng(limitations_seed + 2)
            # Seed sticky keys RNG  
            self._rng_sticky = np.random.default_rng(limitations_seed + 3)

    def main(self, draw, draw_value=False, save_values=False, max_score=200, track_rewards=False): 

        #Initialize pygame screen if wanted
        if draw:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_icon(bird_image)
            pygame.display.set_caption('Flappy Bird')
            clock = pygame.time.Clock()
            
            # Initialize font for Q-value display
            if draw_value:
                font = pygame.font.Font(None, 36)

        #Initialize game
        active_episode = True
        self.init_game()
        
        # Initialize reward tracking for evaluation
        if track_rewards:
            self.tracked_rewards = []
        
        # Initialize data collection for saving values
        if save_values:
            values_data = {
                'episode_data': [],
                'metadata': {
                    'agent_type': type(self.agent).__name__,
                    'timestamp': datetime.now().isoformat(),
                    'device': self.device if hasattr(self, 'device') else 'unknown'
                }
            }
            step_count = 0

        #Game loop
        while active_episode:

            if draw:
                clock.tick(30)
                screen.blit(BACKGROUND, (0, 0))

                #Check for closing game window
                for event in pygame.event.get():
                    if event.type == QUIT:
                        active_episode = False

            # Ensure PPO runs in eval mode during inference to disable dropout
            if isinstance(self.agent, ppo_agent.PPO_agent):
                self.agent.model.eval()

            #Get and execute agent action
            state = self.game_state()
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
            
            if executed_action == 1: 
                self.bird.bump()
                # Set cooldown period after successful jump (random between sticky_keys-2 and sticky_keys+2)
                # Only set cooldown if the action actually succeeded (not blocked by cooldown or failed)
                # Use separate random number generator to avoid affecting pipe generation
                # [STICKY KEYS COOLDOWN] 
                if self.sticky_keys > 0 and not action_blocked_by_cooldown and not action_failed:
                    # min_cooldown = max(1, self.sticky_keys - 2)  # Ensure at least 1 step cooldown
                    # max_cooldown = self.sticky_keys + 2

                    min_cooldown = 0
                    max_cooldown = sticky_keys
                    self.jump_cooldown_remaining = self._rng_sticky.integers(min_cooldown, max_cooldown+1)
            if executed_action == -1: active_episode = False
            
            # Update sticky keys cooldown
            if self.sticky_keys > 0 and self.jump_cooldown_remaining > 0:
                self.jump_cooldown_remaining -= 1

            #Updating environment
            self.bird.update()
            for pipe in self.pipes: pipe.update()
            self.score_update()
            
            # Update goal position in damage mode (move it with the pipes)
            if self.damage_mode and self.goal_x is not None:
                self.goal_x -= GAME_SPEED
            
            # Check for pipe damage in damage mode
            if self.damage_mode:
                self.check_pipe_damage()
            
            #Remove pipes if out of screen and instantiate new ones
            self.pipe_handling()
            
            #Store experience BEFORE checking collision to ensure valid state transitions
            if self.train:
                next_state = self.game_state()
                reward = self.reward(executed_action)
                vars(self.agent)["buffer"][0].append(state)
                vars(self.agent)["buffer"][1].append(next_state)
                vars(self.agent)["buffer"][2].append(reward)
                vars(self.agent)["buffer"][3].append(action)
                # Mark if this is a terminal state (collision occurred or score limit reached)
                is_terminal = self.collision() or self.score >= 200
                vars(self.agent)["buffer"][4].append(is_terminal)
            elif track_rewards:
                # Track rewards even when not training (for evaluation metrics)
                reward = self.reward(executed_action)
                self.tracked_rewards.append(reward)
            
            #Check for collisions
            if self.collision(): active_episode = False

            self.turn += 1
                
            #Update screen
            if draw:
                # In damage mode, draw bird on top of pipes for visibility during collisions
                if self.damage_mode:
                    self.ground.draw(screen)
                    for pipe in self.pipes: pipe.draw(screen)
                    # Draw goal finish line at the final pipes (if visible)
                    if self.goal_x is not None and -PIPE_WIDHT < self.goal_x < SCREEN_WIDHT:
                        screen.blit(goal_image, (self.goal_x, self.goal_y))
                    self.bird.draw(screen)  # Bird on top
                else:
                    self.bird.draw(screen)
                    self.ground.draw(screen)
                    for pipe in self.pipes: pipe.draw(screen)
                pygame.display.update()

            #Terminate episode after reaching target score
            if self.damage_mode:
                # In damage mode, end after passing 10 pipes
                if self.pipes_passed >= self.damage_target_pipes:
                    active_episode = False
            else:
                # Normal mode: end after reaching max_score
                if self.score >= max_score:
                    active_episode = False

        #Quit pygame window
        if draw:
            pygame.display.quit()
            pygame.quit()
        
        # In damage mode, return damage taken as the "score"
        if self.damage_mode:
            if self.verbose:
                print(f"Episode ended: {self.pipes_passed} pipes passed, {self.damage_taken} damage taken")
            return self.damage_taken
        
        return self.score

        
    def train_agent(self, *args, **kwargs):
        """Training is not part of the stimuli-generation pipeline.

        This codebase now supports PPO inference only; checkpoints should be
        trained outside this trimmed pipeline and placed in models/.
        """
        raise NotImplementedError("Training is not supported in this PPO-only generation pipeline.")

    def _test_and_save_best_model(self, best_model_state, latest_model_state, batches, save_path, best_model_episodes, latest_model_episode, best_mean_score, use_training_limitations=True):
        """Test both models and save the better one"""
        if self.verbose:
            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            
            # Print model information
            if best_model_episodes:
                episode_number = best_model_episodes[0]
                print(f"Best model so far: Achieved mean score of {best_mean_score:.2f} at episode {episode_number}")
            else:
                print("Best model so far: No episodes tracked")
                
            print(f"Latest model: From episode {latest_model_episode}")
            print("-" * 60)
        
        # Test both models
        best_mean, _ = self._test_model(best_model_state, 50, "Best model so far", use_training_limitations)
        latest_mean, _ = self._test_model(latest_model_state, 50, "Latest model", use_training_limitations)
        
        if self.verbose:
            print("-" * 60)
            # Save the better model
            if best_mean >= latest_mean:
                self._load_and_save_model(best_model_state, save_path, best_mean, "best model so far")
                print(f"✅ Selected: Best model so far (episode {best_model_episodes[0]})")
            else:
                self._load_and_save_model(latest_model_state, save_path, latest_mean, "latest model")
                print(f"✅ Selected: Latest model (episode {latest_model_episode})")
            
            print(f"Model saved to: {save_path}")
            print("="*60)
        else:
            # Save the better model without verbose output
            if best_mean >= latest_mean:
                self._load_and_save_model(best_model_state, save_path, best_mean, "best model so far")
            else:
                self._load_and_save_model(latest_model_state, save_path, latest_mean, "latest model")

    def _test_model(self, model_state, batches, model_name, use_training_limitations=True, verbose=True):
        """Test a model and return (mean_score, mean_reward_per_step)"""
        # Store original model states to restore later (deep copy for safety)
        original_model_state = copy.deepcopy(self.agent.model.state_dict())
        original_target_model_state = None
        if hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
            original_target_model_state = copy.deepcopy(self.agent.target_model.state_dict())
        original_training_steps = self.agent.training_steps
        original_train_flag = self.train
        
        # Load the saved model state properly
        if isinstance(model_state, dict) and 'model_state' in model_state:
            # New format with complete state
            self.agent.model.load_state_dict(model_state['model_state'])
            if 'target_model_state' in model_state and hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
                self.agent.target_model.load_state_dict(model_state['target_model_state'])
            if 'training_steps' in model_state:
                self.agent.training_steps = model_state['training_steps']
        else:
            # Legacy format - only main model state available
            # Load into main model and sync target (this breaks target network relationship)
            self.agent.model.load_state_dict(model_state)
            if hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
                self.agent.target_model.load_state_dict(model_state)
                print(f"  ⚠️  Warning: {model_name} using legacy format - target network sync lost")
        
        # Store original settings
        original_action_fail_probability = self.action_fail_probability
        original_sticky_keys = self.sticky_keys
        
        # Choose testing conditions
        if use_training_limitations:
            # Keep original training limitations
            pass
        else:
            # Set clean testing conditions (unlimited mode)
            self.action_fail_probability = 0.0
            self.sticky_keys = 0
        self.train = False
        
        # Track rewards across all test episodes
        all_rewards = []
        
        try:
            scores = []
            for _ in range(batches):
                # Run episode with reward tracking enabled
                score = self.main(draw=False, track_rewards=True)
                scores.append(score)
                
                # Collect rewards from this episode
                if hasattr(self, 'tracked_rewards') and self.tracked_rewards:
                    all_rewards.extend(self.tracked_rewards)
            
            mean_score = np.mean(scores)
            mean_reward_per_step = np.mean(all_rewards) if all_rewards else 0.0
            
            if verbose:
                print(f"{model_name}: mean score = {mean_score:.2f} (scores: {scores}), mean reward/step = {mean_reward_per_step:.4f}")
            
            return mean_score, mean_reward_per_step
        finally:
            # Restore original settings and model states
            self.action_fail_probability = original_action_fail_probability
            self.sticky_keys = original_sticky_keys
            self.train = original_train_flag
            
            # Restore original model states
            self.agent.model.load_state_dict(original_model_state)
            if hasattr(self.agent, 'target_model') and original_target_model_state is not None:
                self.agent.target_model.load_state_dict(original_target_model_state)
            self.agent.training_steps = original_training_steps

    def _load_and_save_model(self, model_state, save_path, score, model_type):
        """Load and save a model"""
        # Load the saved model state properly
        if isinstance(model_state, dict) and 'model_state' in model_state:
            # New format with complete state
            self.agent.model.load_state_dict(model_state['model_state'])
            if 'target_model_state' in model_state and hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
                self.agent.target_model.load_state_dict(model_state['target_model_state'])
            if 'training_steps' in model_state:
                self.agent.training_steps = model_state['training_steps']
        else:
            # Legacy format - only main model state available
            self.agent.model.load_state_dict(model_state)
            if hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
                self.agent.target_model.load_state_dict(model_state)
        
        self.agent.save_model(save_path)
        if self.verbose:
            print(f"✅ Saving {model_type} (score: {score:.2f})")

    def _snapshot_agent_state(self):
        """Create a deep copy snapshot of the agent's model (and target) state for safe testing/saving."""
        snapshot = {
            'model_state': copy.deepcopy(self.agent.model.state_dict()),
            'training_steps': getattr(self.agent, 'training_steps', 0)
        }
        # Preserve auxiliary model state if the active PPO implementation exposes one.
        if hasattr(self.agent, 'target_model') and getattr(self.agent, 'target_model') is not None:
            snapshot['target_model_state'] = copy.deepcopy(self.agent.target_model.state_dict())
        return snapshot
