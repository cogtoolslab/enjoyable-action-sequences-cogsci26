"""
Map Difficulty Analysis for Flappy Bird

Evaluates the difficulty of each map from outputs/ by running a PPO agent
multiple times on each map in headless mode (no visualization). Difficulty is 
measured as completion rate (percentage of times the agent survives all 10 pipes).

Returns top 5 most difficult and easiest maps.

Optimized for speed:
- Runs pure simulation without any rendering
- Supports parallel processing across maps (use --parallel flag)
- On multi-core systems, parallel mode provides near-linear speedup

Usage:
    # Serial mode (default)
    python 04_score_difficulty.py --num-trials 200
    
    # Parallel mode (recommended for large datasets)
    python 04_score_difficulty.py --num-trials 200 --parallel
    
    # Parallel with custom worker count
    python 04_score_difficulty.py --num-trials 200 --parallel --workers 8
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import functools
import os

# Add generation pipeline directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# game.py loads assets at import time using relative paths.
_original_cwd = os.getcwd()
try:
    os.chdir(PROJECT_ROOT)
    from core.game import Game
    from core.config import PIPE_DISTANCE, PIPE_GAP, SCREEN_HEIGHT, PIPE_HEIGHT, PIPE_WIDHT, GROUND_HEIGHT, BIRD_WIDTH, BIRD_HEIGHT, GROUND_WIDHT
    from core import objects
finally:
    os.chdir(_original_cwd)

import pygame

# Models to use for map difficulty assessment
VALUE_MODELS = ['value_ppo_mr_2_1_on_10.pth', 'value_ppo_mr_3_1.5_on_15.pth', 'value_ppo_mr_5_2_on_20.pth']

# Difficulty as number of collisions
# Outputs: for each model, in the same file: model, motor_response, obs_noise_std, num_trials, total_maps, all_results
# all_results: for each map: filename, map_number, completion_rate, nb_collisions, stats

# Agent configuration (based on model name: mr_5_2 = motor response mean=5, std=2)
MOTOR_RESPONSE = (3.0, 1.5)
MOTOR_RESPONSES = [(2.0, 1.0), (3.0, 1.5), (5.0, 2.0)]
OBS_NOISE_STD = 15.0
OBS_NOISE_STDS = [10.0, 15.0, 20.0]
NUM_TRIALS = 100
DEFAULT_OUTPUT_DIR_NAMES = [
    "mr_2_1_10_high_amplitude",
    "mr_2_1_10_low_amplitude",
    "mr_5_2_20_high_amplitude",
    "mr_5_2_20_low_amplitude",
    "unlimited_high_amplitude",
    "unlimited_low_amplitude",
]


class GameWithFixedPipes(Game):
    """
    Extended Game class that allows initialization with fixed pipes.
    
    When using fixed pipes, the bird is automatically positioned:
    - Horizontally: at x=70 (standard position, 210 pixels from first pipe)
    - Vertically: centered in the first pipe gap for fair evaluation
    - Speed: 15 (standard starting speed)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_pipes_config = None
        self.use_fixed_pipes = False
    
    def set_fixed_pipes(self, pipes_config: List[Dict]):
        """
        Set a fixed pipe configuration to use on next init_game
        
        Args:
            pipes_config: List of pipe dictionaries with x, y, width, height, inverted
        """
        self.fixed_pipes_config = pipes_config
        self.use_fixed_pipes = True
    
    def pipe_handling(self):
        """
        Override pipe handling to not generate new random pipes when using fixed pipes.
        Only removes pipes that are off-screen.
        """
        if not self.use_fixed_pipes:
            # Use default behavior if not using fixed pipes
            super().pipe_handling()
            return
        
        # For fixed pipes: only remove pipes that are off-screen
        # Do not generate new random pipes
        if len(self.pipes) >= 2 and vars(self.pipes[0])["pos"][0] <= -100:
            del self.pipes[0]
            del self.pipes[0]
    
    def collision(self):
        """
        Override collision detection to handle case where pipes list might be empty.
        """
        # Check ground and roof collision (always fatal)
        if vars(self.bird)["pos"][1] < 0 or vars(self.bird)["pos"][1] > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT:
            return True
        
        # Check for pipe collision only if pipes exist
        if len(self.pipes) == 0:
            return False
        
        # Use parent collision detection
        return super().collision()
    
    def init_game(self):
        """Initialize game, optionally using fixed pipes if set"""
        # Initialize pygame images if not already done (minimal initialization for headless mode)
        if not hasattr(self, '_images_loaded'):
            # We need pygame initialized but we don't need actual images for headless simulation
            # Just create dummy surfaces that match the expected dimensions
            if not pygame.get_init():
                pygame.init()
            self.bird_image = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT))
            self.pipe_image = pygame.Surface((PIPE_WIDHT, PIPE_HEIGHT))
            self.ground_image = pygame.Surface((GROUND_WIDHT, GROUND_HEIGHT))
            self._images_loaded = True
        
        # Initialize game objects
        self.bird = objects.Bird(self.bird_image)
        self.ground = objects.Ground(self.ground_image, 0)
        self.pipes = []
        self.score = 0
        self.turn = 0
        
        # Initialize sticky keys tracking
        self.jump_cooldown_remaining = 0
        
        # Initialize PPO motor-response pending-jump tracking
        self.ppo_jump_pending = False
        self.ppo_pending_delay_remaining = 0
        
        # Initialize damage mode tracking (if applicable)
        if self.damage_mode:
            self.damage_taken = 0
            self.pipes_passed = 0
            self.last_damage_step = -999
            self.is_in_collision = False
            self.collision_ended_at_frame = -999
            self.pipes_created_count = 0
            self.goal_x = PIPE_DISTANCE * self.damage_target_pipes
            self.goal_y = (SCREEN_HEIGHT - GROUND_HEIGHT) // 2
        
        # Initialize pipes - either from fixed config or random
        first_bottom_pipe_y = None
        first_top_pipe_y = None
        
        if self.fixed_pipes_config is not None:
            # Use fixed pipes configuration
            # Note: pipe_config['y'] values are already the actual pos[1] values from the trajectory,
            # not the ysize parameter. We need to reverse-engineer the ysize parameter.
            for pipe_config in self.fixed_pipes_config:
                actual_y = pipe_config['y']
                
                if pipe_config['inverted']:
                    # Top pipe (inverted): pos[1] = -(height - ysize)
                    # So: ysize = height + pos[1] = 500 + actual_y
                    ysize = PIPE_HEIGHT + actual_y
                else:
                    # Bottom pipe: pos[1] = SCREEN_HEIGHT - ysize
                    # So: ysize = SCREEN_HEIGHT - pos[1] = 600 - actual_y
                    ysize = SCREEN_HEIGHT - actual_y
                
                pipe_obj = objects.Pipe(
                    self.pipe_image,
                    pipe_config['inverted'],
                    pipe_config['x'],
                    ysize
                )
                self.pipes.append(pipe_obj)
                
                # Track first pipe pair for bird positioning (using actual pos[1] values)
                if pipe_config['x'] == PIPE_DISTANCE:  # First pipe at x=280
                    if pipe_config['inverted']:
                        first_top_pipe_y = actual_y
                    else:
                        first_bottom_pipe_y = actual_y
                
                # Track pipe creation in damage mode
                if self.damage_mode and not pipe_config['inverted']:
                    self.pipes_created_count += 1
                    if self.pipes_created_count == self.damage_target_pipes:
                        self.goal_y = actual_y
            
            # Position bird vertically centered in first pipe gap
            # Bird X position is already at 70 (default from objects.Bird)
            # Distance to first pipe at x=280 is 210 pixels (same as original game)
            if first_bottom_pipe_y is not None and first_top_pipe_y is not None:
                # Calculate gap boundaries
                # Top pipe: inverted, so its bottom edge is at y + PIPE_HEIGHT
                gap_top = first_top_pipe_y + PIPE_HEIGHT
                # Bottom pipe: normal, so its top edge is at y
                gap_bottom = first_bottom_pipe_y
                
                # Center bird in the gap
                gap_center_y = (gap_top + gap_bottom) / 2
                # Position bird so its center (y + BIRD_HEIGHT/2) aligns with gap center
                vars(self.bird)["pos"][1] = gap_center_y - BIRD_HEIGHT / 2
        else:
            # Use standard random pipe initialization (original behavior)
            import random
            if self.seed is not None:
                random.seed(self.seed)
            
            for i in range(3):
                xpos = PIPE_DISTANCE * i + PIPE_DISTANCE
                ysize = random.randint(200, 300)
                
                self.pipes.append(objects.Pipe(self.pipe_image, False, xpos, ysize))
                self.pipes.append(objects.Pipe(self.pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))
                
                if self.damage_mode:
                    self.pipes_created_count += 1
                    if self.pipes_created_count == self.damage_target_pipes:
                        self.goal_y = ysize


def extract_pipes_from_trajectory(trajectory_path: Path) -> List[Dict]:
    """
    Extract unique pipe PAIRS from a trajectory JSON file in encounter order.
    
    Pipes move left each frame, so we need to identify unique pipe pairs by their
    gap configuration (y-positions), not their x-positions. We preserve the order
    in which pipes were encountered (left to right) by tracking their first 
    appearance in world space.
    
    Args:
        trajectory_path: Path to trajectory JSON file
    
    Returns:
        List of pipe dictionaries with x, y, width, height, inverted (in encounter order)
    """
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    
    # Extract unique pipe pairs by their gap configuration
    # Track first appearance X position in world space for ordering
    # Key: (bottom_pipe_y, top_pipe_y) to identify unique pipe pairs
    unique_pipe_pairs = {}
    
    for step in data['trajectory']:
        if 'pipes' in step:
            pipes = step['pipes']
            step_num = step['step']
            
            # Group pipes into pairs (consecutive bottom + top pipe at same x)
            i = 0
            while i < len(pipes) - 1:
                pipe1 = pipes[i]
                pipe2 = pipes[i + 1]
                
                # Check if these form a pair (same x position, one inverted one not)
                if abs(pipe1['x'] - pipe2['x']) < 5:  # Allow small floating point differences
                    # Identify bottom and top pipe
                    if not pipe1['inverted'] and pipe2['inverted']:
                        bottom_pipe = pipe1
                        top_pipe = pipe2
                    elif pipe1['inverted'] and not pipe2['inverted']:
                        top_pipe = pipe1
                        bottom_pipe = pipe2
                    else:
                        i += 1
                        continue
                    
                    # Calculate world space X position (where pipe actually is in absolute coordinates)
                    # Pipes scroll left by GAME_SPEED each step, so world_x = current_x + (step * GAME_SPEED)
                    world_x = pipe1['x'] + (step_num * 10)  # GAME_SPEED = 10
                    
                    # Use world_x as unique key (rounded to avoid floating point issues)
                    # This ensures we capture all 10 pipes even if some have identical Y positions
                    key = round(world_x / 10) * 10  # Round to nearest 10 pixels
                    
                    # Only record if not seen before (captures first appearance)
                    if key not in unique_pipe_pairs:
                        unique_pipe_pairs[key] = {
                            'bottom_y': bottom_pipe['y'],
                            'top_y': top_pipe['y'],
                            'width': bottom_pipe['width'],
                            'height': bottom_pipe['height'],
                            'first_world_x': world_x  # Track where pipe first appeared
                        }
                    
                    i += 2  # Skip both pipes in the pair
                else:
                    i += 1
    
    # Sort pipes by their first appearance in world space (left to right encounter order)
    sorted_pairs = sorted(unique_pipe_pairs.values(), key=lambda p: p['first_world_x'])
    
    # Convert to list and assign proper sequential x positions
    # Pipes should be positioned at PIPE_DISTANCE * i + PIPE_DISTANCE
    pipe_configs = []
    for i, pair_config in enumerate(sorted_pairs):
        x_pos = PIPE_DISTANCE * (i + 1)  # Start at PIPE_DISTANCE, then 2*PIPE_DISTANCE, etc.
        
        # Bottom pipe
        pipe_configs.append({
            'x': x_pos,
            'y': pair_config['bottom_y'],
            'width': pair_config['width'],
            'height': pair_config['height'],
            'inverted': False
        })
        
        # Top pipe
        pipe_configs.append({
            'x': x_pos,
            'y': pair_config['top_y'],
            'width': pair_config['width'],
            'height': pair_config['height'],
            'inverted': True
        })
    
    return pipe_configs


def evaluate_single_map_worker(args):
    """
    Worker function for parallel map evaluation.
    Must be at module level for pickling.
    
    Args:
        args: Tuple of (traj_file_path, map_index, total_maps, model_path, num_trials, motor_response, obs_noise_std)
    
    Returns:
        Dict with map evaluation results
    """
    traj_file_path, map_idx, total_maps, model_path, num_trials, motor_response, obs_noise_std = args
    
    # Extract filename for reporting
    traj_file = Path(traj_file_path)
    filename = traj_file.name
    
    # No printing in parallel mode - results will be reported at the end
    # This keeps logs clean and avoids jumbled output from concurrent workers
    
    # Extract pipes from trajectory
    pipes_config = extract_pipes_from_trajectory(traj_file)
    
    # Evaluate difficulty (never verbose in parallel mode)
    completion_rate, stats = evaluate_map_difficulty(
        pipes_config,
        model_path,
        num_trials=num_trials,
        motor_response=motor_response,
        obs_noise_std=obs_noise_std,
        verbose=False
    )
    
    # Extract map number
    map_num = extract_map_number(filename)
    
    return {
        'filename': filename,
        'map_number': map_num,
        'completion_rate': completion_rate,
        'nb_collisions': stats['nb_collisions'],
        'stats': stats,
        'map_idx': map_idx  # Include for sorting
    }


def evaluate_map_difficulty(
    pipes_config: List[Dict],
    model_path: str,
    num_trials: int = NUM_TRIALS,
    motor_response: Tuple[float, float] = None,
    obs_noise_std: float = None,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Evaluate map difficulty by running PPO agent multiple times.
    
    Args:
        pipes_config: List of pipe configurations
        model_path: Path to PPO model file
        num_trials: Number of trials to run
        motor_response: Motor response configuration (mean, std). If None, uses default MOTOR_RESPONSE
        obs_noise_std: Observation noise standard deviation. If None, uses default OBS_NOISE_STD
        verbose: Whether to print progress
    
    Returns:
        Tuple of (completion_rate, stats_dict)
        - completion_rate: Percentage of successful completions (0-100)
        - stats_dict: Dict with additional statistics
    """
    # Use provided values or fall back to defaults
    if motor_response is None:
        motor_response = MOTOR_RESPONSE
    if obs_noise_std is None:
        obs_noise_std = OBS_NOISE_STD
    
    # Initialize game with fixed pipes
    game = GameWithFixedPipes(
        agent_name="ppo_agent",
        device="cpu",
        model_path=model_path,
        motor_response=motor_response,
        obs_noise_std=obs_noise_std,
        state_size=5,
        verbose=False,
        damage_mode=False  # Use standard collision detection
    )
    
    # Set the fixed pipes
    game.set_fixed_pipes(pipes_config)
    
    # Track results
    completions = 0
    collisions = 0  # Track number of collisions
    total_scores = []
    survival_steps = []
    
    for trial in range(num_trials):
        if verbose and trial % 100 == 0:
            print(f"  Trial {trial}/{num_trials}...")
        
        # Initialize game with fixed pipes
        game.init_game()
        
        # Debug: print initial state for first trial
        if verbose and trial == 0:
            print(f"  Initial: {len(game.pipes)} pipes, bird at y={vars(game.bird)['pos'][1]:.1f}")
        
        # Run until game over
        steps = 0
        max_steps = 10000  # Safety limit
        active_episode = True
        trial_ended_in_collision = False
        
        while active_episode and steps < max_steps:
            # Check completion first (before state calculation that requires pipes)
            if game.score >= 10:
                active_episode = False
                break
            
            # Check if we still have pipes (needed for state calculation)
            if len(game.pipes) == 0:
                # No more pipes but didn't complete - this is a failure
                active_episode = False
                break
            
            # Check if there are any pipes ahead of the bird (needed for valid state)
            has_pipe_ahead = False
            bird_x = vars(game.bird)["pos"][0]
            for pipe in game.pipes:
                if bird_x - BIRD_WIDTH < vars(pipe)["pos"][0]:
                    has_pipe_ahead = True
                    break
            
            if not has_pipe_ahead:
                # No pipes ahead - agent has passed all pipes
                # Count this as completion if score >= 10, otherwise failure
                active_episode = False
                break
            
            # Get game state
            state = game.game_state()
            
            # Validate state size before passing to agent
            expected_size = 6  # 5 features + 1 pending flag for PPO
            if len(state) != expected_size:
                if verbose:
                    print(f"  Warning: Invalid state size {len(state)}, expected {expected_size}")
                    print(f"    Bird X: {bird_x}, Pipes: {len(game.pipes)}, Score: {game.score}")
                active_episode = False
                break
            
            # Get action from agent
            action = game.agent.act(state, False)
            
            # Determine executed action (considering motor response delays for PPO)
            executed_action = action
            
            # Handle PPO motor response delays
            if game.is_ppo and game.motor_response is not None:
                if game.ppo_jump_pending:
                    # Ignore any new jump selections while pending
                    game.ppo_pending_delay_remaining -= 1
                    if game.ppo_pending_delay_remaining <= 0:
                        executed_action = 1
                        game.ppo_jump_pending = False
                        game.ppo_pending_delay_remaining = 0
                    else:
                        executed_action = 0
                else:
                    if action == 1:
                        import random
                        mean_delay, std_delay = game.motor_response
                        delay = int(max(0, round(random.gauss(mean_delay, std_delay))))
                        if delay <= 0:
                            executed_action = 1
                        else:
                            game.ppo_jump_pending = True
                            game.ppo_pending_delay_remaining = delay
                            executed_action = 0
            
            # Execute action
            if executed_action == 1:
                game.bird.bump()
            elif executed_action == -1:
                active_episode = False
                continue
            
            # Update game objects
            game.bird.update()
            for pipe in game.pipes:
                pipe.update()
            game.score_update()
            game.pipe_handling()
            
            # Check collision
            if game.collision():
                trial_ended_in_collision = True
                active_episode = False
            
            game.turn += 1
            steps += 1
        
        # Check if agent completed all pipes (score >= 10 means passed 10 pipes)
        score = game.score
        total_scores.append(score)
        survival_steps.append(steps)
        
        # Debug: print failure info for first trial
        if verbose and trial == 0:
            print(f"  Trial {trial}: Score={score}, Steps={steps}, Completed={score >= 10}, Collision={trial_ended_in_collision}")
        
        if score >= 10:  # Successfully passed all 10 pipes
            completions += 1
        elif trial_ended_in_collision:  # Trial ended in collision
            collisions += 1
    
    # Calculate statistics
    completion_rate = (completions / num_trials) * 100
    
    stats = {
        'completions': int(completions),
        'nb_collisions': int(collisions),
        'trials': int(num_trials),
        'completion_rate': float(completion_rate),
        'avg_score': float(np.mean(total_scores)),
        'std_score': float(np.std(total_scores)),
        'avg_survival_steps': float(np.mean(survival_steps)),
        'max_score': int(np.max(total_scores)),
        'min_score': int(np.min(total_scores))
    }
    
    return completion_rate, stats


def extract_map_number(filename: str) -> int:
    """Extract map number from filename"""
    try:
        parts = filename.split('map_')
        if len(parts) > 1:
            map_num = parts[1].split('.json')[0]
            return int(map_num)
    except:
        pass
    return -1


def analyze_all_maps(
    outputs_dir: Path = None,
    trajectory_numbers: List[int] = None,
    num_trials: int = NUM_TRIALS,
    save_results: bool = True,
    parallel: bool = False,
    num_workers: int = None,
    model_path: str = None,
    motor_response: Tuple[float, float] = None,
    obs_noise_std: float = None
) -> Dict:
    """
    Analyze difficulty of all maps in outputs directory.
    
    Args:
        outputs_dir: Base directory containing trajectory JSON files or trajectory folders
        trajectory_numbers: Optional list of trajectory numbers to analyze (collects from multiple folders)
        num_trials: Number of trials per map
        save_results: Whether to save results to JSON
        parallel: Whether to use parallel processing
        num_workers: Number of parallel workers (None = auto-detect)
        model_path: Path to model file (if None, uses the first VALUE_MODELS entry)
        motor_response: Motor response configuration (if None, uses default MOTOR_RESPONSE)
        obs_noise_std: Observation noise standard deviation (if None, uses default OBS_NOISE_STD)
    
    Returns:
        Dictionary with analysis results
    """
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent / "outputs"
    
    # Use provided values or fall back to defaults
    if model_path is None:
        model_path = str(Path(__file__).resolve().parent / 'models' / VALUE_MODELS[0])
    if motor_response is None:
        motor_response = MOTOR_RESPONSE
    if obs_noise_std is None:
        obs_noise_std = OBS_NOISE_STD
    
    # Extract model name for display
    model_name = Path(model_path).name
    
    print(f"Analyzing map difficulty...")
    print(f"Model: {model_name}")
    print(f"Motor response: {motor_response}")
    print(f"Obs noise std: {obs_noise_std}")
    print(f"Trials per map: {num_trials}")
    
    # Find all trajectory JSON files
    if trajectory_numbers and len(trajectory_numbers) > 1:
        # Multiple trajectory folders: collect files from all of them
        trajectory_files = []
        for traj_num in trajectory_numbers:
            traj_dir = outputs_dir / str(traj_num)
            if traj_dir.exists():
                files = sorted(traj_dir.glob("*.json"))
                files = [f for f in files if 'danger_analysis' not in f.name and 'map_difficulty' not in f.name]
                trajectory_files.extend(files)
                print(f"  Trajectory {traj_num}: Found {len(files)} files")
        trajectory_files = sorted(trajectory_files)
        print(f"Outputs directories: {[str(outputs_dir / str(t)) for t in trajectory_numbers]}")
    else:
        # Single directory (either base outputs_dir or single trajectory folder)
        if trajectory_numbers and len(trajectory_numbers) == 1:
            outputs_dir = outputs_dir / str(trajectory_numbers[0])
            print(f"Outputs directory: {outputs_dir}")
        else:
            print(f"Outputs directory: {outputs_dir}")
        
        trajectory_files = sorted(outputs_dir.glob("*.json"))
        trajectory_files = [f for f in trajectory_files if 'danger_analysis' not in f.name and 'map_difficulty' not in f.name]
    
    print(f"\nFound {len(trajectory_files)} trajectory files total")
    
    # Analyze each map
    results = []
    
    if parallel:
        # Parallel processing mode
        if num_workers is None:
            num_workers = max(1, cpu_count() - 4)  # Leave 4 cores free
        
        print(f"Using parallel processing with {num_workers} workers")
        print(f"Processing {len(trajectory_files)} maps...")
        
        # Prepare arguments for each worker
        worker_args = [
            (str(traj_file), i, len(trajectory_files), model_path, num_trials, motor_response, obs_noise_std)
            for i, traj_file in enumerate(trajectory_files)
        ]
        
        # Process maps in parallel with progress tracking
        with Pool(num_workers) as pool:
            # Use imap_unordered for progress tracking
            results = []
            for i, result in enumerate(pool.imap_unordered(evaluate_single_map_worker, worker_args), 1):
                results.append(result)
                # Print progress every 50 maps
                if i % 50 == 0 or i == len(trajectory_files):
                    print(f"  ✓ Completed {i}/{len(trajectory_files)} maps ({100*i//len(trajectory_files)}%)")
        
        print(f"\n{'='*70}")
        print("✓ Parallel processing complete!")
        print(f"{'='*70}\n")
        
        # Print summary of first few results (sorted by original index)
        print("Sample results (first 5 maps):")
        sorted_results = sorted(results, key=lambda x: x['map_idx'])
        for result in sorted_results[:5]:
            print(f"  Map {result['map_number']}: {result['completion_rate']:.1f}% completion")
        print()
        
        # Remove map_idx from results (only needed for sorting)
        for result in results:
            result.pop('map_idx', None)
        
    else:
        # Serial processing mode (original)
        for i, traj_file in enumerate(trajectory_files):
            # Print progress every 10 maps to reduce I/O overhead
            if (i+1) % 10 == 0 or i < 5:
                print(f"[{i+1}/{len(trajectory_files)}] Analyzing {traj_file.name}...")
            
            # Extract pipes from trajectory
            pipes_config = extract_pipes_from_trajectory(traj_file)
            
            # Debug: show pipe count for first few maps
            if i < 3:
                num_pairs = len(pipes_config) // 2
                print(f"  Extracted {len(pipes_config)} pipes ({num_pairs} pairs)")
            
            # Evaluate difficulty
            completion_rate, stats = evaluate_map_difficulty(
                pipes_config,
                model_path,
                num_trials=num_trials,
                motor_response=motor_response,
                obs_noise_std=obs_noise_std,
                verbose=(i < 1)  # Very verbose for first map only
            )
            
            if i % 10 == 0 or i < 5:
                print(f"  Completion: {completion_rate:.1f}% | Avg score: {stats['avg_score']:.2f} ± {stats['std_score']:.2f}")
            
            # Extract map number
            map_num = extract_map_number(traj_file.name)
            
            # Store results
            result = {
                'filename': traj_file.name,
                'map_number': map_num,
                'completion_rate': completion_rate,
                'nb_collisions': stats['nb_collisions'],
                'stats': stats
            }
            results.append(result)
    
    # Sort by completion rate (primary), then by avg score (secondary for ties)
    # For hardest: lower completion rate is harder, then lower avg score is harder
    # For easiest: higher completion rate is easier, then higher avg score is easier
    results_sorted = sorted(results, key=lambda x: (x['completion_rate'], x['stats']['avg_score']))
    
    # Get top 5 hardest (lowest completion rate, then lowest avg score) and easiest (highest completion rate, then highest avg score)
    hardest_maps = results_sorted[:5]
    easiest_maps = results_sorted[-5:][::-1]  # Reverse to show highest first
    
    # Prepare output
    analysis_results = {
        'model': model_name,
        'motor_response': motor_response,
        'obs_noise_std': obs_noise_std,
        'num_trials': num_trials,
        'total_maps': len(results),
        'all_results': results_sorted,
        'hardest_maps': hardest_maps,
        'easiest_maps': easiest_maps
    }
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nTop 5 HARDEST Maps (Lowest Completion Rate):")
    print("-" * 70)
    for i, result in enumerate(hardest_maps, 1):
        print(f"{i}. Map {result['map_number']}: {result['completion_rate']:.1f}% "
              f"(avg score: {result['stats']['avg_score']:.2f})")
        print(f"   {result['filename']}")
    
    print("\nTop 5 EASIEST Maps (Highest Completion Rate):")
    print("-" * 70)
    for i, result in enumerate(easiest_maps, 1):
        print(f"{i}. Map {result['map_number']}: {result['completion_rate']:.1f}% "
              f"(avg score: {result['stats']['avg_score']:.2f})")
        print(f"   {result['filename']}")
    
    # Save results
    if save_results:
        # If multiple trajectory numbers, save to base outputs directory
        if trajectory_numbers and len(trajectory_numbers) > 1:
            save_dir = Path(__file__).parent / "outputs"
            output_file = save_dir / "map_difficulty_results.json"
        else:
            output_file = outputs_dir / "map_difficulty_results.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return analysis_results


def analyze_all_maps_multiple_models(
    outputs_dir: Path = None,
    trajectory_numbers: List[int] = None,
    num_trials: int = NUM_TRIALS,
    save_results: bool = True,
    parallel: bool = False,
    num_workers: int = None
) -> Dict:
    """
    Analyze difficulty of all maps using multiple value models sequentially.
    Each model uses its corresponding motor response and obs noise by index.
    
    Args:
        outputs_dir: Base directory containing trajectory JSON files or trajectory folders
        trajectory_numbers: Optional list of trajectory numbers to analyze (collects from multiple folders)
        num_trials: Number of trials per map
        save_results: Whether to save results to JSON
        parallel: Whether to use parallel processing
        num_workers: Number of parallel workers (None = auto-detect)
    
    Returns:
        Dictionary with all model results
    """
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent / "outputs"
    
    # Ensure we have matching lists
    num_models = len(VALUE_MODELS)
    if len(MOTOR_RESPONSES) != num_models:
        raise ValueError(f"Number of motor responses ({len(MOTOR_RESPONSES)}) doesn't match number of models ({num_models})")
    if len(OBS_NOISE_STDS) != num_models:
        raise ValueError(f"Number of obs noise stds ({len(OBS_NOISE_STDS)}) doesn't match number of models ({num_models})")
    
    print(f"\n{'='*70}")
    print(f"ANALYZING MAPS WITH {num_models} VALUE MODELS")
    print(f"{'='*70}\n")
    
    all_model_results = []
    
    # Run each model sequentially
    for model_idx, model_name in enumerate(VALUE_MODELS):
        print(f"\n{'='*70}")
        print(f"MODEL {model_idx + 1}/{num_models}: {model_name}")
        print(f"{'='*70}")
        
        # Get corresponding motor response and obs noise
        motor_response = MOTOR_RESPONSES[model_idx]
        obs_noise_std = OBS_NOISE_STDS[model_idx]
        
        # Get model path
        model_path = str(Path(__file__).resolve().parent / 'models' / model_name)
        
        # Analyze maps with this model
        model_results = analyze_all_maps(
            outputs_dir=outputs_dir,
            trajectory_numbers=trajectory_numbers,
            num_trials=num_trials,
            save_results=False,  # Don't save individual model results
            parallel=parallel,
            num_workers=num_workers,
            model_path=model_path,
            motor_response=motor_response,
            obs_noise_std=obs_noise_std
        )
        
        all_model_results.append(model_results)
    
    # Prepare combined output
    combined_results = {
        'model_results': all_model_results
    }
    
    # Save combined results
    if save_results:
        # If multiple trajectory numbers, save to base outputs directory
        if trajectory_numbers and len(trajectory_numbers) > 1:
            save_dir = Path(__file__).parent / "outputs"
            output_file = save_dir / "map_difficulty_results.json"
        else:
            output_file = outputs_dir / "map_difficulty_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"✓ All results saved to: {output_file}")
        print(f"{'='*70}")
    
    # Print summary for all models
    print(f"\n{'='*70}")
    print("SUMMARY FOR ALL MODELS")
    print(f"{'='*70}\n")
    
    for model_idx, model_result in enumerate(all_model_results):
        print(f"Model {model_idx + 1}: {model_result['model']}")
        print(f"  Motor response: {model_result['motor_response']}")
        print(f"  Obs noise std: {model_result['obs_noise_std']}")
        print(f"  Total maps: {model_result['total_maps']}")
        if model_result['all_results']:
            avg_completion = np.mean([r['completion_rate'] for r in model_result['all_results']])
            print(f"  Avg completion rate: {avg_completion:.1f}%")
        print()
    
    return combined_results



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Flappy Bird map difficulty")
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default=None,
        help='Directory containing trajectory JSON files'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=NUM_TRIALS,
        help=f'Number of trials per map (default: {NUM_TRIALS})'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to JSON'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing across maps (much faster on multi-core systems)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--trajectory-number', '-t',
        type=int,
        nargs='+',
        default=None,
        help='Trajectory number(s) to analyze (can specify multiple, e.g., -t 441 12 352)'
    )
    parser.add_argument(
        '--single-model',
        action='store_true',
        help='Run analysis with single model only (default: run all models sequentially)'
    )
    args = parser.parse_args()
    
    # Set outputs directory
    if args.outputs_dir:
        output_dirs = [Path(__file__).parent / "outputs" / args.outputs_dir]
        trajectory_numbers = None  # Custom dir overrides trajectory numbers
    elif args.trajectory_number:
        output_dirs = [Path(__file__).parent / "outputs"]
        trajectory_numbers = args.trajectory_number
    else:
        output_dirs = [
            Path(__file__).parent / "outputs" / dirname
            for dirname in DEFAULT_OUTPUT_DIR_NAMES
        ]
        trajectory_numbers = None
    
    for output_dir in output_dirs:
        print(f"\n{'='*70}")
        print(f"DIFFICULTY ANALYSIS INPUT: {output_dir}")
        print(f"{'='*70}")

        # Run analysis - use multi-model by default
        if args.single_model:
            analyze_all_maps(
                outputs_dir=output_dir,
                trajectory_numbers=trajectory_numbers,
                num_trials=args.num_trials,
                save_results=not args.no_save,
                parallel=args.parallel,
                num_workers=args.workers
            )
        else:
            analyze_all_maps_multiple_models(
                outputs_dir=output_dir,
                trajectory_numbers=trajectory_numbers,
                num_trials=args.num_trials,
                save_results=not args.no_save,
                parallel=args.parallel,
                num_workers=args.workers
            )
    
    print("\n✓ Analysis complete!")
