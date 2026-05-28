"""
Random Map Generator from Trajectory

This script generates N random collision-free maps from an input trajectory by:
1. Loading the original trajectory and extracting action sequences
2. Incrementally generating pipe pairs one at a time
3. Validating each pipe by simulating bird passage with original actions
4. Rejecting and regenerating pipes that cause collisions
5. Saving collision-free trajectories to output directory

Input: Trajectory JSON file (e.g., analysis/trajectories/*.json)
Output: N JSON files in outputs/
"""

import json
import random
import os
import sys
from pathlib import Path

# Add generation pipeline directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_INITIAL_TRAJECTORIES = [
    SCRIPT_DIR / "initial_trajectories" / "mr_2_1_10_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_2_1_10_trajectory_low_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_5_2_20_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_5_2_20_trajectory_low_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "unlimited_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "unlimited_trajectory_low_amplitude.json",
]
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
from core.config import *


def resolve_path(path_value):
    """Resolve paths from cwd first, then from the generation pipeline root."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    for candidate in (Path.cwd() / path, PROJECT_ROOT / path, SCRIPT_DIR / path):
        if candidate.exists() or candidate.parent.exists():
            return candidate

    return PROJECT_ROOT / path


def load_trajectory(trajectory_path):
    """Load trajectory JSON and extract action sequence"""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    
    # Extract action sequence from trajectory
    actions = [step['action'] for step in data['trajectory']]
    
    return data, actions


def generate_random_pipe_pair(pipe_index, is_initial=False):
    """
    Generate a random pipe pair (bottom + top)
    
    Args:
        pipe_index: Index of the pipe pair (0-9)
        is_initial: Whether this is one of the initial 3 pipes
    
    Returns:
        tuple: (bottom_pipe_dict, top_pipe_dict)
    """
    # Determine x position
    if is_initial:
        # Initial 3 pipes at positions 280, 560, 840
        xpos = PIPE_DISTANCE * (pipe_index + 1)
    else:
        # Subsequent pipes will be positioned during simulation
        xpos = None  # Will be set during simulation
    
    # Random height for bottom pipe
    if is_initial:
        ysize = random.randint(200, 350)
    else:
        ysize = random.randint(150, 400)
    
    # Bottom pipe (non-inverted)
    bottom_pipe = {
        "x": xpos,
        "y": SCREEN_HEIGHT - ysize,
        "width": PIPE_WIDHT,
        "height": PIPE_HEIGHT,
        "inverted": False,
        "ysize": ysize  # Store for reference
    }
    
    # Top pipe (inverted)
    top_pipe = {
        "x": xpos,
        "y": -(PIPE_HEIGHT - ysize),
        "width": PIPE_WIDHT,
        "height": PIPE_HEIGHT,
        "inverted": True,
        "ysize": ysize  # Same ysize as bottom
    }
    
    return bottom_pipe, top_pipe


def create_bird_state():
    """Initialize bird state matching game.py initialization"""
    return {
        "x": 70.0,
        "y": 317.0,
        "speed": 17.0,
        "width": BIRD_WIDTH,
        "height": BIRD_HEIGHT
    }


def update_bird_physics(bird, action):
    """
    Update bird physics for one step
    
    Args:
        bird: Bird state dict
        action: 0 (do nothing) or 1 (jump)
    """
    # Apply action (jump if action == 1)
    if action == 1:
        bird["speed"] = -SPEED
    
    # Apply gravity and update position
    bird["speed"] += GRAVITY
    bird["y"] += bird["speed"]


def update_pipes(pipes):
    """Move all pipes left by GAME_SPEED"""
    for pipe in pipes:
        pipe["x"] -= GAME_SPEED


def check_collision(bird, pipes):
    """
    Check if bird collides with ground, ceiling, or pipes
    
    Returns:
        tuple: (collision_occurred, collision_type)
        collision_type: 'ground', 'ceiling', 'pipe', or None
    """
    # Check ground collision
    if bird["y"] > SCREEN_HEIGHT - GROUND_HEIGHT - bird["height"]:
        return True, 'ground'
    
    # Check ceiling collision
    if bird["y"] < 0:
        return True, 'ceiling'
    
    # Check pipe collisions
    for i in range(0, len(pipes), 2):  # Process pairs (bottom, top)
        bottom_pipe = pipes[i]
        
        # Check horizontal overlap
        if bottom_pipe["x"] - bird["width"] < bird["x"] < bottom_pipe["x"] + bottom_pipe["width"]:
            # Check vertical collision
            pipe_y = bottom_pipe["y"]  # Bottom edge of bottom pipe
            
            # Bird collides if:
            # 1. Bird's bottom hits bottom pipe: pipe_y < bird_y + bird_height
            # 2. Bird's top hits top pipe: pipe_y - PIPE_GAP > bird_y
            if pipe_y < bird["y"] + bird["height"] or pipe_y - PIPE_GAP > bird["y"]:
                return True, 'pipe'
    
    return False, None


def bird_passed_pipe(bird, pipe_x):
    """Check if bird has passed a pipe"""
    return bird["x"] > pipe_x + PIPE_WIDHT


def calculate_game_state(bird, pipes):
    """
    Calculate normalized game state (6 features)
    
    Returns:
        list: [horizontal_dist, vertical_dist, speed, ground_dist, ceiling_dist, pending_flag]
    """
    # Find next pipe (first pipe in front of bird)
    next_pipe = None
    for i in range(0, len(pipes), 2):
        pipe = pipes[i]
        if bird["x"] - bird["width"] < pipe["x"]:
            next_pipe = pipe
            break
    
    if next_pipe is None:
        # No pipes ahead (shouldn't happen in normal gameplay)
        return [0.0, 0.0, bird["speed"] / SPEED, 0.0, 0.0, 0.0]
    
    # Calculate features
    horizontal_distance = (next_pipe["x"] - bird["x"] - bird["width"]) / PIPE_DISTANCE
    vertical_distance = (next_pipe["y"] - bird["y"] - bird["height"]) / SCREEN_HEIGHT
    normalized_speed = bird["speed"] / SPEED
    ground_distance = (SCREEN_HEIGHT - GROUND_HEIGHT - bird["y"] - bird["height"]) / SCREEN_HEIGHT
    ceiling_distance = bird["y"] / SCREEN_HEIGHT
    pending_flag = 0.0  # Always 0 for non-PPO simulation
    
    return [
        horizontal_distance,
        vertical_distance,
        normalized_speed,
        ground_distance,
        ceiling_distance,
        pending_flag
    ]


def get_pipes_snapshot(pipes):
    """Create a snapshot of current pipes for trajectory recording"""
    return [{
        "x": p["x"],
        "y": p["y"],
        "width": p["width"],
        "height": p["height"],
        "inverted": p["inverted"]
    } for p in pipes]


def simulate_full_trajectory(actions, pipe_heights, num_pipes=10):
    """
    Simulate full trajectory with given pipe heights
    
    Args:
        actions: List of actions from original trajectory
        pipe_heights: List of ysize values for each of the num_pipes pipe pairs
        num_pipes: Total number of pipe pairs (default 10)
    
    Returns:
        tuple: (success, trajectory, collision_info)
        - success: True if completed without collision
        - trajectory: List of trajectory step dicts
        - collision_info: dict with 'pipe_index' and 'type' if collision occurred
    """
    # Initialize bird
    bird = create_bird_state()
    
    # Initialize first 3 pipes with provided heights
    pipes = []
    for i in range(3):
        xpos = PIPE_DISTANCE * (i + 1)
        ysize = pipe_heights[i]
        
        bottom_pipe = {
            "x": xpos,
            "y": SCREEN_HEIGHT - ysize,
            "width": PIPE_WIDHT,
            "height": PIPE_HEIGHT,
            "inverted": False,
            "pipe_index": i
        }
        # Top pipe ysize must account for the gap
        ysize_top = SCREEN_HEIGHT - ysize - PIPE_GAP
        top_pipe = {
            "x": xpos,
            "y": -(PIPE_HEIGHT - ysize_top),
            "width": PIPE_WIDHT,
            "height": PIPE_HEIGHT,
            "inverted": True,
            "pipe_index": i
        }
        pipes.extend([bottom_pipe, top_pipe])
    
    pipes_created = 3
    pipes_passed = 0
    trajectory = []
    
    for step, action in enumerate(actions):
        # Record state before action
        game_state = calculate_game_state(bird, pipes)
        trajectory_step = {
            "step": step,
            "time": 0.0,
            "bird": {
                "x": bird["x"],
                "y": bird["y"],
                "speed": bird["speed"]
            },
            "action": action,
            "game_state": game_state,
            "pipes": get_pipes_snapshot(pipes),
            "score": pipes_passed,
            "collision": False,
            "damage_taken": 0,
            "pipes_passed": pipes_passed
        }
        trajectory.append(trajectory_step)
        
        # Execute action and update physics
        update_bird_physics(bird, action)
        update_pipes(pipes)
        
        # Check for collisions
        collision, collision_type = check_collision(bird, pipes)
        if collision:
            # Determine which pipe caused collision
            if collision_type == 'pipe':
                # Find the pipe we collided with
                for i in range(0, len(pipes), 2):
                    bottom_pipe = pipes[i]
                    if bottom_pipe["x"] - bird["width"] < bird["x"] < bottom_pipe["x"] + bottom_pipe["width"]:
                        collision_info = {
                            'pipe_index': bottom_pipe["pipe_index"],
                            'type': collision_type
                        }
                        return False, trajectory, collision_info
            else:
                # Ground or ceiling collision
                collision_info = {'pipe_index': None, 'type': collision_type}
                return False, trajectory, collision_info
        
        # Check if bird passed first pipe (score update)
        if len(pipes) >= 2 and bird["x"] == pipes[0]["x"]:
            pipes_passed += 1
        
        # Handle pipe generation (remove old, add new)
        if len(pipes) >= 2 and pipes[0]["x"] <= -100:
            # Remove old pipes
            del pipes[0]
            del pipes[0]
            
            # Add new pipe if we haven't created all pipes yet
            if pipes_created < num_pipes:
                xpos = PIPE_DISTANCE * 3 - 100
                ysize = pipe_heights[pipes_created]
                
                bottom_pipe = {
                    "x": xpos,
                    "y": SCREEN_HEIGHT - ysize,
                    "width": PIPE_WIDHT,
                    "height": PIPE_HEIGHT,
                    "inverted": False,
                    "pipe_index": pipes_created
                }
                # Top pipe ysize must account for the gap
                ysize_top = SCREEN_HEIGHT - ysize - PIPE_GAP
                top_pipe = {
                    "x": xpos,
                    "y": -(PIPE_HEIGHT - ysize_top),
                    "width": PIPE_WIDHT,
                    "height": PIPE_HEIGHT,
                    "inverted": True,
                    "pipe_index": pipes_created
                }
                pipes.extend([bottom_pipe, top_pipe])
                pipes_created += 1
        
        # Check if we've passed all pipes
        if pipes_passed >= num_pipes:
            return True, trajectory, None
    
    # Completed trajectory without collision
    if pipes_passed >= num_pipes:
        return True, trajectory, None
    else:
        # Ran out of actions
        collision_info = {'pipe_index': None, 'type': 'out_of_actions'}
        return False, trajectory, collision_info


def generate_collision_free_map(actions, num_pipes=10, max_retries_per_pipe=10000):
    """
    Generate a complete collision-free map using incremental pipe generation.
    Generates and validates one pipe pair at a time.
    
    Args:
        actions: List of actions from original trajectory
        num_pipes: Number of pipe pairs to generate (default 10)
        max_retries_per_pipe: Max attempts per pipe before giving up
    
    Returns:
        tuple: (success, full_trajectory, pipe_heights_used)
    """
    # Start with empty list of validated pipe heights
    validated_pipe_heights = []
    
    # Generate and validate each pipe one at a time
    for pipe_index in range(num_pipes):
        pipe_validated = False
        attempts = 0
        current_pipe_height = None
        
        while not pipe_validated and attempts < max_retries_per_pipe:
            attempts += 1
            
            # Generate a random height for the current pipe
            if pipe_index < 2:
                current_pipe_height = random.randint(200, 300)
            else:
                current_pipe_height = random.randint(150, 350)
            
            # Build complete pipe_heights list for testing:
            # - Use validated heights for pipes 0 to pipe_index-1
            # - Use current_pipe_height for pipe_index
            # - Use placeholder random heights for future pipes (so we can simulate full trajectory)
            pipe_heights = validated_pipe_heights.copy()
            pipe_heights.append(current_pipe_height)
            
            # Add placeholder heights for future pipes
            for future_idx in range(pipe_index + 1, num_pipes):
                if future_idx < 3:
                    pipe_heights.append(random.randint(200, 300))
                else:
                    pipe_heights.append(random.randint(150, 350))
            
            # Simulate full trajectory with this pipe configuration
            success, trajectory, collision_info = simulate_full_trajectory(
                actions, pipe_heights, num_pipes
            )
            
            if success:
                # All pipes validated! Current pipe is good
                validated_pipe_heights.append(current_pipe_height)
                pipe_validated = True
                # Return the successful trajectory
                return True, trajectory, pipe_heights
            else:
                # Collision occurred
                if collision_info['type'] in ['ground', 'ceiling', 'out_of_actions']:
                    # Catastrophic failure - can't complete with these early pipes
                    print(f"  Map failed: {collision_info['type']} collision")
                    return False, None, None
                elif collision_info['pipe_index'] is not None:
                    collision_pipe_idx = collision_info['pipe_index']
                    
                    if collision_pipe_idx == pipe_index:
                        # Collision with the pipe we're currently validating - regenerate it
                        # Continue loop to retry with new random height
                        continue
                    elif collision_pipe_idx < pipe_index:
                        # Collision with a previously validated pipe - this shouldn't happen
                        print(f"  ERROR: Collision with already-validated pipe {collision_pipe_idx + 1}")
                        return False, None, None
                    else:
                        # Collision with a future pipe (placeholder) - this means we successfully
                        # passed the current pipe. The current pipe is validated.
                        # We'll handle the future pipe collision when we validate that pipe.
                        validated_pipe_heights.append(current_pipe_height)
                        pipe_validated = True
                        break
        
        if not pipe_validated:
            print(f"  Failed to validate pipe {pipe_index + 1} after {max_retries_per_pipe} attempts")
            return False, None, None
    
    # Should not reach here (should return on success or failure above)
    return False, None, None


def save_trajectory(trajectory, output_path):
    """Save trajectory to JSON file (without metadata)"""
    output_data = {
        "trajectory": trajectory
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def process_single_trajectory(input_trajectory_path, num_maps_to_generate, output_root=DEFAULT_OUTPUT_ROOT):
    """
    Process a single trajectory and generate maps for it.
    
    Args:
        input_trajectory_path: Path to trajectory JSON file
        num_maps_to_generate: Number of maps to generate for this trajectory
        output_root: Directory where the per-trajectory output folder is created
    """
    input_trajectory_path = resolve_path(input_trajectory_path)
    output_root = resolve_path(output_root)
    trajectory_name = input_trajectory_path.stem.replace("_trajectory_", "_")
    output_dir = output_root / trajectory_name
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    print(f"\n{'='*70}")
    print(f"Processing trajectory: {input_trajectory_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    trajectory_data, actions = load_trajectory(input_trajectory_path)
    
    # Extract base filename for output
    base_filename = input_trajectory_path.stem  # Remove .json
    
    # Generate N random maps
    print(f"\nGenerating {num_maps_to_generate} collision-free maps...")
    for map_idx in range(num_maps_to_generate):
        print(f"Map {map_idx + 1}/{num_maps_to_generate}")
        
        success = False
        map_attempts = 0
        max_map_attempts = 10
        
        while not success and map_attempts < max_map_attempts:
            map_attempts += 1
            if map_attempts > 1:
                print(f"  Retrying map generation (attempt {map_attempts})...")
            success, trajectory, pipes = generate_collision_free_map(actions)
            
            if success:
                # Save trajectory
                output_filename = f"{base_filename}_map_{map_idx}.json"
                output_path = output_dir / output_filename
                save_trajectory(trajectory, output_path)
            else:
                print(f"  ✗ Map generation failed")
        
        if not success:
            print(f"  ERROR: Could not generate map {map_idx + 1} after {max_map_attempts} attempts")
            return False
    
    print(f"\n✓ Successfully generated {num_maps_to_generate} maps for trajectory {trajectory_name}!")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate collision-free map variants from seed trajectory JSON files."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[str(path) for path in DEFAULT_INITIAL_TRAJECTORIES],
        help="One or more seed trajectory JSON files. Defaults to the six bundled seed trajectories.",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=1000,
        help="Number of map variants to generate per seed trajectory (default: 1000).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for generated map folders (default: outputs).",
    )
    args = parser.parse_args()

    input_trajectory_paths = args.input
    num_maps_to_generate = args.num_maps
    
    # Process each trajectory
    total_trajectories = len(input_trajectory_paths)
    print(f"\n{'='*70}")
    print(f"MAP GENERATION FOR {total_trajectories} TRAJECTORY(IES)")
    print(f"{'='*70}")
    
    for traj_idx, input_trajectory_path in enumerate(input_trajectory_paths, 1):
        print(f"\n[{traj_idx}/{total_trajectories}] Processing trajectory...")
        
        success = process_single_trajectory(
            input_trajectory_path,
            num_maps_to_generate,
            output_root=args.output_root,
        )
        
        if not success:
            print(f"\n✗ Failed to process trajectory {traj_idx}: {input_trajectory_path}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Completed processing {total_trajectories} trajectory(ies)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
