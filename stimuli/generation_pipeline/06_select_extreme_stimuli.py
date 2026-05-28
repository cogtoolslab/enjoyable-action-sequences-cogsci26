"""
High Danger and Difficulty Stimuli Generation

This script generates a single 10-pipe-pair map that meets both:
1. Danger score (mean over 3 value models) > or < than danger_threshold
2. Completion rate (mean over 3 models × 100 trials = 300 total) > or < than completion_threshold

Uses iterative generation with max_attempts to find a suitable map.
"""

import json
import sys
import os
import signal
import time
import threading
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np
from multiprocessing import Manager, Pool, cpu_count

# Add generation pipeline directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_TRAJECTORIES = [
    SCRIPT_DIR / "initial_trajectories" / "mr_2_1_10_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_2_1_10_trajectory_low_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_5_2_20_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "mr_5_2_20_trajectory_low_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "unlimited_trajectory_high_amplitude.json",
    SCRIPT_DIR / "initial_trajectories" / "unlimited_trajectory_low_amplitude.json",
]
DEFAULT_OUTPUTS_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_STIMULI_OUTPUT_DIR = DEFAULT_OUTPUTS_ROOT / "stimuli_trajectories"

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


# Import functions from existing modules using importlib
import importlib.util

# Import map creation helpers
spec1 = importlib.util.spec_from_file_location(
    "map_creation",
    Path(__file__).parent / "03_create_maps.py"
)
map_creation = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(map_creation)

# Import difficulty scoring helpers
spec2 = importlib.util.spec_from_file_location(
    "map_analysis",
    Path(__file__).parent / "04_score_difficulty.py"
)
map_analysis = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(map_analysis)

# Import danger scoring helpers
spec3 = importlib.util.spec_from_file_location(
    "danger_analysis",
    Path(__file__).parent / "05_score_danger.py"
)
danger_analysis = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(danger_analysis)

# Import shared visual rendering helpers
spec4 = importlib.util.spec_from_file_location(
    "render_visuals",
    Path(__file__).parent / "07_render_visuals.py"
)
render_visuals = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(render_visuals)
trajectory_viz = render_visuals
video_generation = render_visuals

# Import model_utils for danger computation
from utils.model_utils import instantiate_agent, align_state_to_size

# Model configurations
# Note: Danger analysis uses optimal_value models, map analysis uses regular models
# We use the optimal_value models for danger, and regular models for difficulty
DANGER_VALUE_MODELS = danger_analysis.VALUE_MODELS  # optimal_value models
DIFFICULTY_VALUE_MODELS = map_analysis.VALUE_MODELS  # regular models
MOTOR_RESPONSES = danger_analysis.MOTOR_RESPONSES
OBS_NOISE_STDS = danger_analysis.OBS_NOISE_STDS


def generate_single_map(actions, num_pipes=10):
    """
    Generate a single collision-free map.
    
    Args:
        actions: List of actions from original trajectory
        num_pipes: Number of pipe pairs to generate (default 10)
    
    Returns:
        tuple: (success, trajectory, pipe_heights)
    """
    return map_creation.generate_collision_free_map(actions, num_pipes=num_pipes)


def extract_pipes_from_trajectory_data(trajectory_data: List[Dict]) -> List[Dict]:
    """
    Extract unique pipe pairs from trajectory data structure (in-memory).
    Adapted from extract_pipes_from_trajectory() in 04_score_difficulty.py
    
    Args:
        trajectory_data: List of trajectory step dictionaries
    
    Returns:
        List of pipe dictionaries with x, y, width, height, inverted (in encounter order)
    """
    unique_pipe_pairs = {}
    
    for step in trajectory_data:
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
                    
                    # Calculate world space X position
                    world_x = pipe1['x'] + (step_num * 10)  # GAME_SPEED = 10
                    key = round(world_x / 10) * 10  # Round to nearest 10 pixels
                    
                    # Only record if not seen before
                    if key not in unique_pipe_pairs:
                        unique_pipe_pairs[key] = {
                            'bottom_y': bottom_pipe['y'],
                            'top_y': top_pipe['y'],
                            'width': bottom_pipe['width'],
                            'height': bottom_pipe['height'],
                            'first_world_x': world_x
                        }
                    
                    i += 2  # Skip both pipes in the pair
                else:
                    i += 1
    
    # Sort pipes by their first appearance in world space
    sorted_pairs = sorted(unique_pipe_pairs.values(), key=lambda p: p['first_world_x'])
    
    # Convert to list and assign proper sequential x positions
    pipe_configs = []
    for i, pair_config in enumerate(sorted_pairs):
        x_pos = PIPE_DISTANCE * (i + 1)
        
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


def evaluate_map_danger_multi_model(trajectory_data: List[Dict]) -> float:
    """
    Evaluate map danger using mean of 3 value models.
    
    Args:
        trajectory_data: List of trajectory steps containing game_state
    
    Returns:
        Mean danger score across 3 value models
    """
    danger_scores = []
    
    for model_name in DANGER_VALUE_MODELS:
        # Create a fresh agent cache for each model
        agent_cache = {}
        
        # Compute danger for this model
        danger_score = danger_analysis.compute_danger(
            trajectory_data,
            model=model_name,
            agent_cache=agent_cache
        )
        danger_scores.append(danger_score)
    
    return np.mean(danger_scores)


def evaluate_map_difficulty_multi_model(trajectory_data: List[Dict], num_trials=100) -> float:
    """
    Evaluate map difficulty using mean completion rate over 3 models × num_trials.
    
    Args:
        trajectory_data: List of trajectory steps
        num_trials: Number of trials per model (default 100)
    
    Returns:
        Mean completion rate across 3 models
    """
    # Extract pipes from trajectory data
    pipes_config = extract_pipes_from_trajectory_data(trajectory_data)
    
    completion_rates = []
    
    for model_idx, model_name in enumerate(DIFFICULTY_VALUE_MODELS):
        # Get model-specific configurations
        motor_response = MOTOR_RESPONSES[model_idx]
        obs_noise_std = OBS_NOISE_STDS[model_idx]
        
        # Get model path
        model_path = str(Path(__file__).resolve().parent / 'models' / model_name)
        
        # Evaluate difficulty with this model
        completion_rate, _ = map_analysis.evaluate_map_difficulty(
            pipes_config,
            model_path,
            num_trials=num_trials,
            motor_response=motor_response,
            obs_noise_std=obs_noise_std,
            verbose=False
        )
        
        completion_rates.append(completion_rate)
    
    return np.mean(completion_rates)


def _worker_generate_and_evaluate(args):
    """
    Worker function for parallel map generation + evaluation loop.
    
    Args:
        args: Tuple of (
            worker_id, actions, thresholds, output_dir_str, trajectory_name,
            num_trials, max_attempts_per_worker, shared_corners_found,
            shared_attempt_counter, shared_lock
        )
    Returns:
        Dict of results for corners found by this worker:
        {corner_name: (success, output_path, stats_dict)}
    """
    (
        worker_id,
        actions,
        thresholds,
        output_dir_str,
        trajectory_name,
        num_trials,
        max_attempts_per_worker,
        shared_corners_found,
        shared_attempt_counter,
    ) = args
    
    danger_threshold_high = thresholds["danger_high"]
    danger_threshold_low = thresholds["danger_low"]
    completion_threshold_high = thresholds["completion_high"]
    completion_threshold_low = thresholds["completion_low"]
    
    output_dir = Path(output_dir_str)
    local_results: Dict[str, Tuple[bool, str, Dict]] = {}
    
    for attempt in range(1, max_attempts_per_worker + 1):
        # Early exit if all corners already found
        if all(shared_corners_found[corner] for corner in shared_corners_found.keys()):
            break
        
        # Generate map
        success, trajectory, pipe_heights = generate_single_map(actions, num_pipes=10)
        if not success:
            # Increment counter even for failed generations (atomic update)
            shared_attempt_counter["count"] = shared_attempt_counter["count"] + 1
            continue
        
        # Evaluate danger and difficulty
        danger_score = evaluate_map_danger_multi_model(trajectory)
        completion_rate = evaluate_map_difficulty_multi_model(trajectory, num_trials=num_trials)
        
        # Log progress every 100 attempts (atomic update)
        shared_attempt_counter["count"] = shared_attempt_counter["count"] + 1
        total_attempts = shared_attempt_counter["count"]
        if total_attempts % 100 == 0:
            found_count = sum(shared_corners_found[corner] for corner in shared_corners_found.keys())
            print(f"Attempt {total_attempts}: Danger={danger_score:.4f}, Completion={completion_rate:.2f}% ({found_count}/4 corners found)")
        
        # Determine which corner (if any) this map belongs to
        corner_matched = None
        danger_dir = None
        completion_dir = None
        
        if (
            not shared_corners_found["low_low"]
            and danger_score < danger_threshold_low
            and completion_rate < completion_threshold_low
        ):
            corner_matched = "low_low"
            danger_dir = "low"
            completion_dir = "low"
        elif (
            not shared_corners_found["low_high"]
            and danger_score < danger_threshold_low
            and completion_rate > completion_threshold_high
        ):
            corner_matched = "low_high"
            danger_dir = "low"
            completion_dir = "high"
        elif (
            not shared_corners_found["high_low"]
            and danger_score > danger_threshold_high
            and completion_rate < completion_threshold_low
        ):
            corner_matched = "high_low"
            danger_dir = "high"
            completion_dir = "low"
        elif (
            not shared_corners_found["high_high"]
            and danger_score > danger_threshold_high
            and completion_rate > completion_threshold_high
        ):
            corner_matched = "high_high"
            danger_dir = "high"
            completion_dir = "high"
        
        if corner_matched is None:
            continue
        
        # Attempt to claim this corner (atomic check+set via shared dict semantics)
        if shared_corners_found[corner_matched]:
            # Another worker already claimed it in the meantime
            continue
        
        # Mark as found
        shared_corners_found[corner_matched] = True
        
        # Save trajectory and artifacts
        difficulty_dir = "low" if completion_dir == "high" else "high"
        output_filename = f"{trajectory_name}_{danger_dir}_danger_{danger_score:.3f}_{difficulty_dir}_difficulty_{(100-completion_rate):.2f}.json"
        output_path = output_dir / output_filename
        
        map_creation.save_trajectory(trajectory, output_path)
        
        # Visualization
        viz_filename = output_filename.replace(".json", ".png")
        viz_path = output_dir / viz_filename
        try:
            trajectory_viz.create_static_visualization(
                str(output_path),
                output_path=str(viz_path),
                sample_rate=10,
            )
        except Exception:
            viz_path = None
        
        # Video: generate 4 color variants (red, blue, yellow, green)
        video_path = None
        try:
            color_sprites = {
                "red": "bird-upflap-1.png",
                "blue": "bird-upflap-2.png",
                "yellow": "bird-upflap-3.png",
                "green": "bird-upflap-4.png",
            }
            generated_any = False
            for color, sprite_name in color_sprites.items():
                generated_videos = video_generation.generate_videos_from_trajectories(
                    trajectory_paths=[str(output_path)],
                    output_dir=str(output_dir),
                    fps=24,
                    bird_sprite=sprite_name,
                    filename_suffix=color,
                )
                if generated_videos and not generated_any:
                    # Keep the first generated video path (e.g. red) for stats
                    video_path = Path(generated_videos[0])
                    generated_any = True
        except Exception:
            video_path = None
        
        stats = {
            "danger_score": float(danger_score),
            "completion_rate": float(completion_rate),
            "danger_threshold_low": float(danger_threshold_low),
            "danger_threshold_high": float(danger_threshold_high),
            "completion_threshold_low": float(completion_threshold_low),
            "completion_threshold_high": float(completion_threshold_high),
            "corner": corner_matched,
            "attempts": attempt,
            "pipe_heights": pipe_heights,
            "visualization_path": str(viz_path) if viz_path is not None else None,
            "video_path": str(video_path) if video_path is not None else None,
        }
        
        local_results[corner_matched] = (True, str(output_path), stats)
    
    return local_results


def generate_danger_difficulty_map(
    trajectory_path: str,
    danger_threshold_high: float,
    danger_threshold_low: float,
    completion_threshold_high: float,
    completion_threshold_low: float,
    max_attempts: int = 10000,
    num_trials: int = 100,
    num_workers: int = None,
    output_dir: Path = DEFAULT_STIMULI_OUTPUT_DIR,
) -> Dict:
    """
    Generate 4 maps, one for each corner of the danger/completion space, using
    parallel workers where each worker runs an independent generation + evaluation loop.
    """
    # Load actions from trajectory (shared across workers, read-only)
    trajectory_path = resolve_path(trajectory_path)
    _, actions = map_creation.load_trajectory(trajectory_path)
    
    # Extract base filename for output
    trajectory_name = Path(trajectory_path).stem
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a Manager context so all managed objects and their server
    # process are cleanly shut down after each trajectory.
    with Manager() as manager:
        # Shared state across workers
        shared_corners_found = manager.dict({
            "low_low": False,
            "low_high": False,
            "high_low": False,
            "high_high": False,
        })
        # Shared counter for total attempts (used only for progress logging)
        shared_attempt_counter = manager.dict({"count": 0})
        
        thresholds = {
            "danger_high": float(danger_threshold_high),
            "danger_low": float(danger_threshold_low),
            "completion_high": float(completion_threshold_high),
            "completion_low": float(completion_threshold_low),
        }
        
        # Determine number of workers and attempts per worker
        if num_workers is None:
            num_workers = min(max(1, cpu_count() - 2), 4)
        else:
            num_workers = max(1, num_workers)
        attempts_per_worker = max(1, (max_attempts + num_workers - 1) // num_workers)
        
        worker_args = [
            (
                worker_id,
                actions,
                thresholds,
                str(output_dir),
                trajectory_name,
                num_trials,
                attempts_per_worker,
                shared_corners_found,
                shared_attempt_counter,
            )
            for worker_id in range(num_workers)
        ]
        
        results: Dict[str, Tuple[bool, str, Dict]] = {
            "low_low": (False, None, None),
            "low_high": (False, None, None),
            "high_low": (False, None, None),
            "high_high": (False, None, None),
        }
        
        print(f"\n{'='*70}")
        print(f"PARALLEL GENERATION: {num_workers} workers, {attempts_per_worker} attempts/worker")
        print(f"{'='*70}\n")
        
        pool = Pool(processes=num_workers)
        worker_results_list = []
        
        try:
            worker_results_list = pool.map(_worker_generate_and_evaluate, worker_args)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            # Get worker PIDs before terminate (in case terminate hangs)
            worker_pids = []
            try:
                pool_workers = getattr(pool, '_pool', [])
                for worker in pool_workers:
                    if hasattr(worker, 'pid') and worker.pid:
                        worker_pids.append(worker.pid)
            except Exception:
                pass
            
            # Use threading to timeout terminate() if it hangs
            terminate_completed = threading.Event()
            terminate_exception = [None]
            
            def do_terminate():
                try:
                    pool.terminate()  # Kill all workers immediately
                    terminate_exception[0] = None
                except Exception as e:
                    terminate_exception[0] = e
                finally:
                    terminate_completed.set()
            
            terminate_thread = threading.Thread(target=do_terminate, daemon=True)
            terminate_thread.start()
            
            # Wait up to 2 seconds for terminate to complete
            terminate_timeout = 2.0
            if not terminate_completed.wait(timeout=terminate_timeout):
                # Terminate timed out, forcefully kill workers
                for pid in worker_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # Already dead
                    except Exception:
                        pass
                # Give them a moment to die, then send SIGKILL if needed
                time.sleep(0.5)
                for pid in worker_pids:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        os.kill(pid, signal.SIGKILL)  # Force kill
                    except ProcessLookupError:
                        pass  # Already dead
                    except Exception:
                        pass  # Ignore errors
            
            # Check if terminate() completed successfully
            terminate_succeeded = terminate_completed.is_set() and terminate_exception[0] is None
            
            if terminate_succeeded:
                # Join with timeout using threading
                join_completed = threading.Event()
                join_exception = [None]
                
                def do_join():
                    try:
                        pool.join()
                        join_exception[0] = None
                    except Exception as e:
                        join_exception[0] = e
                    finally:
                        join_completed.set()
                
                join_thread = threading.Thread(target=do_join, daemon=True)
                join_thread.start()
                
                # Wait up to 2 seconds for join to complete
                join_completed.wait(timeout=2.0)
        
        # Merge worker results; keep the first successful result per corner
        for worker_results in worker_results_list:
            for corner, value in worker_results.items():
                if value is None:
                    continue
                success, output_path, stats = value
                if success and not results[corner][0]:
                    results[corner] = (success, output_path, stats)
        
        # Summary
        found_count = sum(1 for corner, (success, _, _) in results.items() if success)
        if found_count < 4:
            print(f"\n{'='*70}")
            print(f"⚠ Found {found_count}/4 corner maps after {max_attempts} attempts (total across workers)")
            print(f"{'='*70}")
            for corner, (success, _, _) in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {corner}")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS! Found all 4 corner maps in parallel")
            print(f"{'='*70}\n")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a high danger and difficulty map"
    )

    parser.add_argument(
        '--max-attempts',
        type=int,
        default=10000,
        help='Maximum number of map generation attempts (default: 10000)'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=100,
        help='Number of trials per model for difficulty evaluation (default: 100)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers to use (default: cpu_count() - 2, capped at 4)'
    )
    parser.add_argument(
        '--trajectory-path',
        nargs='+',
        default=[str(path) for path in DEFAULT_TRAJECTORIES],
        help='One or more seed trajectory JSON files to use. Defaults to the six bundled seed trajectories.'
    )
    parser.add_argument(
        '--outputs-root',
        default=str(DEFAULT_OUTPUTS_ROOT),
        help='Root directory containing map analysis output folders.'
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_STIMULI_OUTPUT_DIR),
        help='Directory for selected stimuli JSON, PNG, and MP4 files.'
    )
    
    args = parser.parse_args()
    
    trajectory_paths = [resolve_path(path) for path in args.trajectory_path]
    outputs_root = resolve_path(args.outputs_root)
    output_dir = resolve_path(args.output_dir)
    
    for trajectory_path in trajectory_paths:
        print(f"\n{'='*100}")
        print(f"PROCESSING TRAJECTORY: {Path(trajectory_path).stem}")
        print(f"{'='*100}\n")

        #########################################################
        ### Compute 5th and 95th percentiles for danger and difficulty
        #########################################################
        outputs_dir = outputs_root / Path(trajectory_path).stem.replace("_trajectory_", "_")
        difficulty_file = outputs_dir / 'map_difficulty_results.json'
        danger_file = outputs_dir / 'danger_analysis_results.json'

        # Load map difficulty results
        with open(difficulty_file, 'r') as f:
            difficulty_data = json.load(f)

        # Load danger analysis results
        with open(danger_file, 'r') as f:
            danger_data = json.load(f)
        
        # Create dataframes from the results
        difficulty_all_results = [item for v in list(difficulty_data.values())[0] for item in v['all_results']]
        difficulty_df_ungrouped = pd.DataFrame(difficulty_all_results)
        difficulty_df = difficulty_df_ungrouped.groupby('map_number').agg({
            'filename': 'first',
            'completion_rate': 'mean',
            'nb_collisions': 'mean',
        }).reset_index()

        danger_all_results = [item for v in list(danger_data.values())[0] for item in v['all_results']]
        danger_df_ungrouped = pd.DataFrame(danger_all_results)
        danger_df = danger_df_ungrouped.groupby('map_number').agg({
            'filename': 'first',
            'danger_score': 'mean',
        }).reset_index()

        # Merge the dataframes on map_number
        merged_df = pd.merge(
            difficulty_df[['map_number', 'filename', 'completion_rate']],
            danger_df[['map_number', 'danger_score']],
            on='map_number',
            how='inner'
        )

        # Calculate 5th and 95th percentiles for danger_score
        danger_high = merged_df['danger_score'].quantile(0.95)
        danger_low = merged_df['danger_score'].quantile(0.05)

        # Calculate 5th and 95th percentiles for completion_rate
        completion_high = merged_df['completion_rate'].quantile(0.95)
        completion_low = merged_df['completion_rate'].quantile(0.05)

        print(f"Danger thresholds: low < {danger_low:.4f} < high < {danger_high:.4f}")
        print(f"Completion thresholds: low < {completion_low:.2f}% < high < {completion_high:.2f}%")
        print(f"\n")

        #########################################################

        # Generate all 4 corner maps (in parallel inside this call)
        results = generate_danger_difficulty_map(
            trajectory_path=trajectory_path,
            danger_threshold_high=danger_high,
            danger_threshold_low=danger_low,
            completion_threshold_high=completion_high,
            completion_threshold_low=completion_low,
            max_attempts=args.max_attempts,
            num_trials=args.num_trials,
            num_workers=args.num_workers,
            output_dir=output_dir,
        )

    exit(0)


if __name__ == "__main__":
    main()
