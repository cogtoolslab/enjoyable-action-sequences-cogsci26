# Models to use to generate seed trajectories.
AGENT_MODEL_NAMES = [
    'value_ppo_unlimited.pth',
    'value_ppo_mr_2_1_on_10.pth',
    'value_ppo_mr_5_2_on_20.pth',
]

# Motor response and observation noise settings for each model.
MODEL_SETTINGS = {
    'value_ppo_unlimited.pth': {'motor_response': None, 'obs_noise_std': 0.0},
    'value_ppo_mr_2_1_on_10.pth': {'motor_response': (2.0, 1.0), 'obs_noise_std': 10.0},
    'value_ppo_mr_5_2_on_20.pth': {'motor_response': (5.0, 2.0), 'obs_noise_std': 20.0},
}

NUM_TRIALS = 10



import sys
import os
import json
from pathlib import Path
import random
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count, Value, Lock
import time
import signal

# This script lives in the generation pipeline root.
project_root = Path(__file__).resolve().parent

# Convert to string for os.path operations
project_root_str = str(project_root)

# Add to path
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.model_utils import get_model_config


class suppress_native_stderr:
    """Temporarily silence C-library stderr output, such as libpng profile warnings."""

    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self._stderr_fd, 2)
        os.close(self._stderr_fd)
        os.close(self._devnull_fd)


with suppress_native_stderr():
    from recording import game_recorder

TARGET_PIPES = 10  # Must reach pipe pair 10 (like generate_damage_stimuli.py)
MAX_ATTEMPTS = 100000  # Safety limit

# Parallelization configuration
NUM_WORKERS = None  # None = use cpu_count(), set to a number to override
USE_PARALLEL = True  # Set to False to run sequentially (for debugging)

AGENT_MODEL = None
AGENT_MODEL_NAME = None
MOTOR_RESPONSE = None
OBS_NOISE_STD = 0.0
OUTPUT_DIR = None
model_path = None
base_state_size = None
agent_name = None


def model_name_from_filename(agent_model):
    return agent_model.replace('value_ppo_', '').replace('_on_', '_').replace('.pth', '')


def configure_model(agent_model):
    """Configure globals used by the trajectory collection helpers."""
    global AGENT_MODEL, AGENT_MODEL_NAME, MOTOR_RESPONSE, OBS_NOISE_STD
    global OUTPUT_DIR, model_path, base_state_size, agent_name

    AGENT_MODEL = agent_model
    AGENT_MODEL_NAME = model_name_from_filename(agent_model)

    settings = MODEL_SETTINGS.get(agent_model, {})
    MOTOR_RESPONSE = settings.get('motor_response')
    OBS_NOISE_STD = settings.get('obs_noise_std', 0.0)

    OUTPUT_DIR = project_root / 'initial_trajectories' / AGENT_MODEL_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(project_root_str, 'models', AGENT_MODEL)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    cfg = get_model_config(model_path)
    model_type = str(cfg.get('model_type', 'ppo'))
    if model_type != 'ppo':
        raise ValueError(f"Unsupported model type: {model_type}. This pipeline only supports PPO checkpoints.")
    input_size = int(cfg.get('state_size', 5))

    base_state_size = input_size
    if model_type == 'ppo' and input_size in (4, 6):
        base_state_size = input_size - 1
    if base_state_size not in (3, 5):
        base_state_size = 5

    agent_name = 'ppo_agent'

    print(f"Project root: {project_root_str}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {AGENT_MODEL}")
    print(f"Motor response: {MOTOR_RESPONSE}")
    print(f"Observation noise std: {OBS_NOISE_STD}")
    print(f"Target: {NUM_TRIALS} trajectories reaching pipe pair {TARGET_PIPES} (collisions allowed)")
    if USE_PARALLEL:
        num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count() - 2
        print(f"Parallel mode: {num_workers} workers")
    else:
        print("Sequential mode (single worker)")


def create_game_instance():
    """Create a new game instance (used by workers)"""
    with suppress_native_stderr():
        rg = game_recorder.RecorderGame(
            agent_name,
            'cpu',
            model_path=model_path,
            action_fail_probability=0.0,
            sticky_keys=0,
            state_size=base_state_size,
            verbose=False,
            seed=0,  # Will be overridden per episode
            stop_at_score=None,  # Not used in damage mode
            np_seed=None,
            motor_response=MOTOR_RESPONSE if agent_name == 'ppo_agent' else None,
            obs_noise_std=OBS_NOISE_STD,
            damage_mode=True,  # Enable damage mode to continue after collisions
            damage_target_pipes=TARGET_PIPES,  # Stop at pipe pair 10
        )
    # Ensure evaluation mode
    if hasattr(rg, 'agent') and hasattr(rg.agent, 'model'):
        rg.agent.model.eval()
        if hasattr(rg.agent, 'target_model') and rg.agent.target_model is not None:
            rg.agent.target_model.eval()
    return rg

def run_episode_and_get_trajectory(rg, seed=None, np_seed=None, quick_check=False):
    """
    Run one episode and return trajectory data.
    
    Args:
        rg: Game instance to use (created per worker)
        seed: Random seed for the game (None for random)
        np_seed: NumPy RNG seed for limitations (motor response, etc.) - must be same for quick check and full recording
        quick_check: If True, run without recording (faster for quick checks)
    
    Returns:
        Tuple of (trajectory_data dict, pipes_passed, damage_taken, success)
        success is True if pipes_passed >= TARGET_PIPES (regardless of collisions)
    """
    if seed is None:
        seed = random.randint(0, 1000000)
    
    # Generate deterministic np_seed for limitations if not provided
    # This ensures motor response delays are deterministic
    if np_seed is None:
        # Use seed to generate a deterministic np_seed
        np_seed = (seed * 1000003) & 0x7FFFFFFF
    
    # Reset game state for this episode
    rg.seed = seed
    rg.np_seed = np_seed  # Use deterministic seed for limitations
    rg.step_count = 0
    rg.trajectory_start_time = None
    rg.active = False
    
    # Reset trajectory data
    rg.trajectory_data = {
        'metadata': {
            'game_seed': seed,
            'agent_type': type(rg.agent).__name__,
            'timestamp': datetime.now().isoformat(),
            'recording_start_time': None,
            'recording_end_time': None,
            'total_steps': 0,
            'final_score': 0,
            'pipes_passed': 0,
            'damage_taken': 0,
            'success': False,
            'duration_seconds': 0.0
        },
        'trajectory': []
    }
    
    # Run the game
    if quick_check:
        # Quick check without recording
        rg.main(draw=False, save_values=False, record_frames=False, max_score=None)
        pipes_passed = int(rg.pipes_passed)
        damage_taken = int(rg.damage_taken)
        success = pipes_passed >= TARGET_PIPES
        return None, pipes_passed, damage_taken, success
    else:
        # Full run with trajectory recording
        # Note: record_frames=True is needed to capture trajectory data (capture_trajectory_step 
        # is only called when record_frames=True). However, we don't need the expensive frame data.
        rg.main(draw=False, save_values=False, record_frames=True, max_score=None)
        
        # Clear the expensive frame data we don't need (saves memory)
        # The trajectory data is already captured in rg.trajectory_data
        rg.recording_data['episode_data'] = []
        
        # Get trajectory data
        trajectory_data = rg.trajectory_data.copy()
        
        # Update metadata
        pipes_passed = int(rg.pipes_passed)
        damage_taken = int(rg.damage_taken)
        trajectory_data['metadata']['final_score'] = pipes_passed  # In damage mode, final_score is pipes_passed
        trajectory_data['metadata']['pipes_passed'] = pipes_passed
        trajectory_data['metadata']['damage_taken'] = damage_taken
        trajectory_data['metadata']['total_steps'] = len(trajectory_data.get('trajectory', []))
        trajectory_data['metadata']['success'] = pipes_passed >= TARGET_PIPES
        
        return trajectory_data, pipes_passed, damage_taken, pipes_passed >= TARGET_PIPES


def process_seed_attempt(args):
    """
    Worker function that processes a single seed attempt.
    Must be at module level for pickling with multiprocessing.
    
    Args:
        args: Tuple of (seed, attempt_number, config_dict)
            - seed: Random seed for the game
            - attempt_number: Attempt number for progress tracking
            - config_dict: Dictionary with configuration (model_path, agent_name, etc.)
    
    Returns:
        Dict with trajectory data and metadata if successful, None otherwise
        Format: {
            'seed': int,
            'attempt': int,
            'trajectory_data': dict,
            'pipes_passed': int,
            'damage_taken': int,
            'success': bool
        }
    """
    seed, attempt_number, config = args
    
    # Unpack configuration
    worker_model_path = config['model_path']
    worker_agent_name = config['agent_name']
    worker_base_state_size = config['base_state_size']
    worker_motor_response = config['motor_response']
    worker_target_pipes = config['target_pipes']
    worker_obs_noise_std = config['obs_noise_std']
    
    # Generate deterministic np_seed for limitations
    np_seed = (seed * 1000003) & 0x7FFFFFFF
    
    # Create game instance for this worker (each process has its own)
    # Use config values instead of global variables
    with suppress_native_stderr():
        rg = game_recorder.RecorderGame(
            worker_agent_name,
            'cpu',
            model_path=worker_model_path,
            action_fail_probability=0.0,
            sticky_keys=0,
            state_size=worker_base_state_size,
            verbose=False,
            seed=0,  # Will be overridden per episode
            stop_at_score=None,  # Not used in damage mode
            np_seed=None,
            motor_response=worker_motor_response,
            obs_noise_std=worker_obs_noise_std,
            damage_mode=True,  # Enable damage mode to continue after collisions
            damage_target_pipes=worker_target_pipes,  # Stop at pipe pair 10
        )
    # Ensure evaluation mode
    if hasattr(rg, 'agent') and hasattr(rg.agent, 'model'):
        rg.agent.model.eval()
        if hasattr(rg.agent, 'target_model') and rg.agent.target_model is not None:
            rg.agent.target_model.eval()
    
    try:
        # Quick check first (faster)
        _, quick_pipes, quick_damage, quick_success = run_episode_and_get_trajectory(
            rg, seed=seed, np_seed=np_seed, quick_check=True
        )
        
        if not quick_success:
            return {
                'seed': seed,
                'attempt': attempt_number,
                'success': False,
                'pipes_passed': quick_pipes,
                'damage_taken': quick_damage,
                'trajectory_data': None
            }
        
        # Quick check passed, now run full recording with same seed and np_seed
        trajectory_data, final_pipes, final_damage, success = run_episode_and_get_trajectory(
            rg, seed=seed, np_seed=np_seed, quick_check=False
        )
        
        if success and trajectory_data is not None:
            # Verify trajectory has data
            trajectory_steps = trajectory_data.get('trajectory', [])
            if len(trajectory_steps) == 0:
                return {
                    'seed': seed,
                    'attempt': attempt_number,
                    'success': False,
                    'pipes_passed': final_pipes,
                    'damage_taken': final_damage,
                    'trajectory_data': None,
                    'error': 'empty_trajectory'
                }
            
            return {
                'seed': seed,
                'attempt': attempt_number,
                'success': True,
                'pipes_passed': final_pipes,
                'damage_taken': final_damage,
                'trajectory_data': trajectory_data,
                'trajectory_length': len(trajectory_steps)
            }
        else:
            return {
                'seed': seed,
                'attempt': attempt_number,
                'success': False,
                'pipes_passed': final_pipes,
                'damage_taken': final_damage,
                'trajectory_data': None,
                'error': 'recording_failed'
            }
    except Exception as e:
        # Handle any errors gracefully
        return {
            'seed': seed,
            'attempt': attempt_number,
            'success': False,
            'pipes_passed': 0,
            'damage_taken': 0,
            'trajectory_data': None,
            'error': str(e)
        }


def collect_trajectories_for_current_model():
    """Generate seed trajectories for the model configured by configure_model()."""
    # Generate N trajectories (all that reach pipe pair 10, regardless of collisions)
    trajectories = []
    damage_counts = []  # Track damage_taken for each trajectory

    print(f"Generating {NUM_TRIALS} trajectories reaching pipe pair {TARGET_PIPES}...")
    print("=" * 60)

    # Prepare configuration for workers
    worker_config = {
        'model_path': model_path,
        'agent_name': agent_name,
        'base_state_size': base_state_size,
        'motor_response': MOTOR_RESPONSE if agent_name == 'ppo_agent' else None,
        'obs_noise_std': OBS_NOISE_STD,
        'target_pipes': TARGET_PIPES
    }

    if USE_PARALLEL:
        # Parallel execution using multiprocessing
        num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count() - 2
        
        # Generate seeds dynamically as needed
        # We'll generate seeds on-demand to avoid memory issues with large NUM_TRIALS
        rng = np.random.RandomState(42)  # Deterministic seed generation
        seed_counter = [0]  # Track how many seeds we've generated
        
        def generate_seed():
            """Generate a new seed deterministically"""
            seed_counter[0] += 1
            # Use the counter to advance the RNG state deterministically
            return int(rng.randint(0, 1000000))
        
        # Shared counters for progress tracking
        attempts_counter = Value('i', 0)
        successful_counter = Value('i', 0)
        attempts_lock = Lock()
        
        # Process seeds in parallel
        start_time = time.time()
        last_progress_time = start_time
        
        pool = Pool(num_workers)
        
        # Submit tasks using apply_async - much simpler and easier to cancel
        # Submit a reasonable number of tasks upfront (enough to keep workers busy)
        pending_tasks = []
        max_pending = num_workers * 3  # Keep 3x workers worth of tasks pending
        
        try:
            # Submit initial batch of tasks
            for i in range(max_pending):
                seed = generate_seed()
                worker_arg = (seed, i + 1, worker_config)
                async_result = pool.apply_async(process_seed_attempt, (worker_arg,))
                pending_tasks.append(async_result)
            
            # Process results as they complete
            # Keep generating until we have NUM_TRIALS trajectories
            while len(trajectories) < NUM_TRIALS:
                # Check for completed tasks
                completed_indices = []
                for i, task in enumerate(pending_tasks):
                    if task.ready():
                        completed_indices.append(i)
                
                # Process completed tasks (in reverse order to maintain indices)
                for i in reversed(completed_indices):
                    task = pending_tasks.pop(i)
                    try:
                        result = task.get(timeout=0.1)  # Should be instant since ready() was True
                    except Exception:
                        continue
                    
                    # Update progress counters
                    with attempts_lock:
                        attempts_counter.value += 1
                        current_attempts = attempts_counter.value
                    
                    # Handle successful results
                    if result['success'] and result['trajectory_data'] is not None:
                        # This is a successful trajectory we'll save
                        with attempts_lock:
                            successful_counter.value += 1
                        
                        seed = result['seed']
                        final_pipes = result['pipes_passed']
                        final_damage = result['damage_taken']
                        trajectory_data = result['trajectory_data']
                        trajectory_length = result['trajectory_length']
                        
                        # Assign trajectory number (sequential)
                        trajectory_number = len(trajectories) + 1
                        
                        # Track damage count for statistics
                        damage_counts.append(final_damage)
                        
                        # Save trajectory using trajectory_number
                        filename = f"{AGENT_MODEL_NAME}_trajectory_{trajectory_number:04d}.json"
                        output_path = OUTPUT_DIR / filename
                        
                        with open(output_path, 'w') as f:
                            json.dump(trajectory_data, f, indent=2)
                        
                        trajectories.append({
                            'index': trajectory_number,
                            'filename': filename,
                            'seed': seed,
                            'pipes_passed': final_pipes,
                            'damage_taken': final_damage,
                            'path': str(output_path),
                            'trajectory_length': trajectory_length
                        })
                        
                        print(f"    ✓ [{trajectory_number}/{NUM_TRIALS}] Saved: {filename} "
                              f"({trajectory_length} steps, {final_damage} collisions)")
                        
                        # Check if we've reached our target
                        if len(trajectories) >= NUM_TRIALS:
                            break
                    
                    # Submit new task if we haven't reached our goal
                    # Generate seeds dynamically - no limit
                    if len(trajectories) < NUM_TRIALS:
                        seed = generate_seed()
                        attempt_num = attempts_counter.value + len(pending_tasks) + 1
                        worker_arg = (seed, attempt_num, worker_config)
                        async_result = pool.apply_async(process_seed_attempt, (worker_arg,))
                        pending_tasks.append(async_result)
                
                # Print periodic progress updates
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:
                    elapsed = current_time - start_time
                    with attempts_lock:
                        current_attempts = attempts_counter.value
                        total_successful = successful_counter.value
                    rate = current_attempts / elapsed if elapsed > 0 else 0
                    saved_count = len(trajectories)
                    print(f"  Progress: {saved_count}/{NUM_TRIALS} saved ({total_successful} successful found), "
                          f"{current_attempts} attempts ({rate:.1f} attempts/sec)")
                    last_progress_time = current_time
                
                # Ensure we maintain enough pending tasks to keep workers busy
                # Submit new tasks if we're below the threshold and haven't reached our goal
                while len(trajectories) < NUM_TRIALS and len(pending_tasks) < max_pending:
                    seed = generate_seed()
                    attempt_num = attempts_counter.value + len(pending_tasks) + 1
                    worker_arg = (seed, attempt_num, worker_config)
                    async_result = pool.apply_async(process_seed_attempt, (worker_arg,))
                    pending_tasks.append(async_result)
                
                # Small sleep to avoid busy-waiting
                if not completed_indices:
                    time.sleep(0.01)
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
            import threading
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
        
        # Get final attempt count
        with attempts_lock:
            total_attempts = attempts_counter.value
        
    else:
        # Sequential execution (original behavior, for debugging)
        attempts = 0
        trajectory_number = 0
        
        # Create a single game instance for sequential mode
        rg = create_game_instance()
        
        while len(trajectories) < NUM_TRIALS and attempts < MAX_ATTEMPTS:
            attempts += 1
            
            # Use a random seed for each attempt
            seed = random.randint(0, 1000000)
            
            # Generate deterministic np_seed for limitations (motor response, etc.)
            np_seed = (seed * 1000003) & 0x7FFFFFFF
            
            # Quick check first (faster) - use same seed and np_seed
            _, quick_pipes, quick_damage, quick_success = run_episode_and_get_trajectory(
                rg, seed=seed, np_seed=np_seed, quick_check=True
            )
            
            if not quick_success:
                if attempts % 100 == 0:
                    print(f"  Attempt {attempts}: pipes={quick_pipes}/{TARGET_PIPES}, damage={quick_damage}")
                continue
            
            # Quick check passed, now run full recording with same seed and np_seed
            print(f"  Attempt {attempts}: pipes={quick_pipes}, damage={quick_damage} - Recording trajectory...")
            
            # Run full recording with the same seed and np_seed (should produce same result)
            trajectory_data, final_pipes, final_damage, success = run_episode_and_get_trajectory(
                rg, seed=seed, np_seed=np_seed, quick_check=False
            )
            
            if success and trajectory_data is not None:
                # Verify trajectory has data
                trajectory_steps = trajectory_data.get('trajectory', [])
                if len(trajectory_steps) == 0:
                    print(f"    ✗ Trajectory is empty (pipes={final_pipes}, damage={final_damage})")
                    continue
                
                # Increment trajectory number only when we successfully save
                trajectory_number += 1
                
                # Track damage count for statistics
                damage_counts.append(final_damage)
                
                # Save trajectory using trajectory_number (not attempt index)
                filename = f"{AGENT_MODEL_NAME}_trajectory_{trajectory_number:04d}.json"
                output_path = OUTPUT_DIR / filename
                
                with open(output_path, 'w') as f:
                    json.dump(trajectory_data, f, indent=2)
                
                trajectories.append({
                    'index': trajectory_number,
                    'filename': filename,
                    'seed': seed,
                    'pipes_passed': final_pipes,
                    'damage_taken': final_damage,
                    'path': str(output_path),
                    'trajectory_length': len(trajectory_steps)
                })
                
                print(f"    ✓ [{trajectory_number}/{NUM_TRIALS}] Saved: {filename} "
                      f"({len(trajectory_steps)} steps, {final_damage} collisions)")
            else:
                print(f"    ✗ Failed to record trajectory (pipes={final_pipes}, damage={final_damage})")
        
        total_attempts = attempts

    print("=" * 60)
    print(f"\n✓ Generated {len(trajectories)}/{NUM_TRIALS} trajectories")
    if total_attempts > 0:
        print(f"  Total attempts: {total_attempts}")
        print(f"  Success rate: {len(trajectories)/total_attempts*100:.1f}%")
    if damage_counts:
        avg_damage = sum(damage_counts) / len(damage_counts)
        print(f"  Average collisions per trajectory: {avg_damage:.2f}")
        print(f"  Total collisions: {sum(damage_counts)}")
        print(f"  Trajectories with 0 collisions: {sum(1 for d in damage_counts if d == 0)}")
        print(f"  Trajectories with collisions: {sum(1 for d in damage_counts if d > 0)}")
    print(f"  Saved to: {OUTPUT_DIR}")


# Main execution - must be wrapped in if __name__ == '__main__' for multiprocessing
if __name__ == '__main__':
    for model_index, agent_model in enumerate(AGENT_MODEL_NAMES, 1):
        print("\n" + "=" * 80)
        print(f"MODEL {model_index}/{len(AGENT_MODEL_NAMES)}: {agent_model}")
        print("=" * 80)
        configure_model(agent_model)
        collect_trajectories_for_current_model()
