"""
Danger Analysis for Flappy Bird Trajectories

Computes the danger of trajectories by evaluating the average V-value
(expected future reward) across all game states in each trajectory.

Lower V-values indicate higher danger (less expected future reward).
Higher V-values indicate lower danger (more expected future reward).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.model_utils import instantiate_agent, align_state_to_size

# Import visualization functions from the same directory
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trajectory_viz",
    Path(__file__).parent / "07_render_visuals.py"
)
trajectory_viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory_viz)

# Models to use for danger assessment
VALUE_MODELS = ['value_ppo_mr_2_1_on_10_optimal_value.pth', 'value_ppo_mr_3_1.5_on_15_optimal_value.pth', 'value_ppo_mr_5_2_on_20_optimal_value.pth']

# Motor response and obs noise configurations (by index, matching VALUE_MODELS)
# Note: These are not used in danger analysis (which only uses value models), but kept for consistency
MOTOR_RESPONSES = [(2.0, 1.0), (3.0, 1.5), (5.0, 2.0)]
OBS_NOISE_STDS = [10.0, 15.0, 20.0]
DEFAULT_OUTPUT_DIR_NAMES = [
    "mr_2_1_10_high_amplitude",
    "mr_2_1_10_low_amplitude",
    "mr_5_2_20_high_amplitude",
    "mr_5_2_20_low_amplitude",
    "unlimited_high_amplitude",
    "unlimited_low_amplitude",
]


def compute_danger(trajectory_data: List[Dict], model: str = None, agent_cache: Dict = None) -> float:
    """
    Compute danger metric for a trajectory.
    
    Danger is computed as the average V-value across all states.
    Lower V-values = higher danger (less expected future reward)
    Higher V-values = lower danger (more expected future reward)
    
    Args:
        trajectory_data: List of trajectory steps containing game_state
        model: Model filename to use for value estimation
        agent_cache: Optional dict to cache loaded agent (avoids reloading)
    
    Returns:
        Average V-value across the trajectory
    """
    # Load or use cached agent
    if agent_cache and 'agent' in agent_cache:
        agent = agent_cache['agent']
    else:
        model_path = str(Path(__file__).resolve().parent / 'models' / model)
        device = 'cpu'
        agent, model_config = instantiate_agent(model_path, device=device, verbose=False)
        if agent_cache is not None:
            agent_cache['agent'] = agent
            agent_cache['config'] = model_config

    # Get value predictions for each game state
    v_values = []

    for step_data in trajectory_data:
        game_state = step_data['game_state']
        
        # Align state to model's expected size
        aligned_state = align_state_to_size(game_state, agent.state_size)
        
        # Get values from the model
        # For PPO: returns [v_value, q_do_nothing, q_jump]
        values = agent.get_q_values(aligned_state)
        
        v_values.append(float(values[0]))

    # Return average V-value
    return 1 - np.mean(v_values)


def load_trajectory(trajectory_path: Path) -> List[Dict]:
    """Load trajectory from JSON file"""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    return data['trajectory']


def extract_map_number(filename: str) -> int:
    """Extract map number from filename (e.g., 'map_42.json' -> 42)"""
    try:
        # Extract the number between 'map_' and '.json'
        parts = filename.split('map_')
        if len(parts) > 1:
            map_num = parts[1].split('.json')[0]
            return int(map_num)
    except:
        pass
    return -1


def analyze_all_trajectories(
    outputs_dir: Path = None, 
    trajectory_numbers: List[int] = None,
    model: str = None
) -> Dict:
    """
    Analyze all trajectories in the outputs directory.
    
    Args:
        outputs_dir: Base directory containing trajectory JSON files or trajectory folders
        trajectory_numbers: Optional list of trajectory numbers to analyze (collects from multiple folders)
        model: Model filename to use (if None, uses the first VALUE_MODELS entry)
    
    Returns:
        Dict with:
            - 'model': Model name used
            - 'results': List of (map_number, filename, danger_score) tuples
            - 'highest_danger': Top 5 highest danger trajectories
            - 'lowest_danger': Top 5 lowest danger trajectories
            - 'stats': Statistics dictionary
    """
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent / 'outputs'
    
    # Use provided model or default
    if model is None:
        model = VALUE_MODELS[0]
    
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
    else:
        # Single directory (either base outputs_dir or single trajectory folder)
        if trajectory_numbers and len(trajectory_numbers) == 1:
            outputs_dir = outputs_dir / str(trajectory_numbers[0])
        
        if not outputs_dir.exists():
            print(f"Error: Outputs directory not found: {outputs_dir}")
            return None
        
        trajectory_files = sorted(outputs_dir.glob("*.json"))
        trajectory_files = [f for f in trajectory_files if 'danger_analysis' not in f.name and 'map_difficulty' not in f.name]
    
    if not trajectory_files:
        print(f"No trajectory files found in {outputs_dir}")
        return None
    
    print(f"Danger Analysis for Flappy Bird Trajectories")
    print("=" * 60)
    print(f"Found {len(trajectory_files)} trajectories to analyze")
    print(f"Using model: {model}")
    print("=" * 60)
    
    # Cache the agent to avoid reloading for each trajectory
    agent_cache = {}
    
    # Compute danger for each trajectory
    results = []
    for idx, traj_path in enumerate(trajectory_files):
        map_number = extract_map_number(traj_path.name)
        
        print(f"Analyzing trajectory {idx + 1}/{len(trajectory_files)}: {traj_path.name}", end='')
        
        try:
            trajectory = load_trajectory(traj_path)
            danger_score = compute_danger(trajectory, model=model, agent_cache=agent_cache)
            results.append((map_number, traj_path.name, danger_score))
            print(f" → Danger: {danger_score:.4f}")
        except Exception as e:
            print(f" → Error: {e}")
            continue
    
    if not results:
        print("No trajectories successfully analyzed")
        return None
    
    # Sort by danger score
    results_sorted = sorted(results, key=lambda x: x[2])
    
    # Get top 5 lowest and highest danger
    lowest_danger = results_sorted[:5]  # Lowest danger = highest V-values
    highest_danger = results_sorted[-5:][::-1]  # Highest danger = lowest V-values
    
    return {
        'model': model,
        'results': results_sorted,
        'highest_danger': highest_danger,
        'lowest_danger': lowest_danger,
        'stats': {
            'mean': np.mean([r[2] for r in results]),
            'std': np.std([r[2] for r in results]),
            'min': min([r[2] for r in results]),
            'max': max([r[2] for r in results]),
        }
    }


def print_results(analysis: Dict):
    """Pretty print the analysis results"""
    if not analysis:
        return
    
    print("\n" + "=" * 60)
    print("DANGER ANALYSIS RESULTS")
    print("=" * 60)
    
    stats = analysis['stats']
    print(f"\nOverall Statistics:")
    print(f"  Mean Danger Score: {stats['mean']:.4f}")
    print(f"  Std Dev:         {stats['std']:.4f}")
    print(f"  Min Danger:        {stats['min']:.4f}")
    print(f"  Max Danger:        {stats['max']:.4f}")
    
    print(f"\n{'TOP 5 HIGHEST DANGER TRAJECTORIES'}")
    print("(Lower V-value = Higher danger = Less expected reward)")
    print("-" * 60)
    print(f"{'Rank':<6} {'Map #':<8} {'Danger Score':<12} {'Filename'}")
    print("-" * 60)
    for rank, (map_num, filename, danger) in enumerate(analysis['highest_danger'], 1):
        print(f"{rank:<6} {map_num:<8} {danger:<12.4f} {filename}")
    
    print(f"\n{'TOP 5 LOWEST DANGER TRAJECTORIES'}")
    print("(Higher V-value = Lower danger = More expected reward)")
    print("-" * 60)
    print(f"{'Rank':<6} {'Map #':<8} {'Danger Score':<12} {'Filename'}")
    print("-" * 60)
    for rank, (map_num, filename, danger) in enumerate(analysis['lowest_danger'], 1):
        print(f"{rank:<6} {map_num:<8} {danger:<12.4f} {filename}")
    
    print("\n" + "=" * 60)
    
    # Print map numbers for easy reference
    highest_maps = [str(m[0]) for m in analysis['highest_danger']]
    lowest_maps = [str(m[0]) for m in analysis['lowest_danger']]
    
    print("\nQuick Reference:")
    print(f"  Highest Danger Maps: {', '.join(highest_maps)}")
    print(f"  Lowest Danger Maps:  {', '.join(lowest_maps)}")
    print("=" * 60)


def visualize_danger_trajectories(analysis: Dict, outputs_dir: Path, trajectory_numbers: List[int] = None):
    """
    Generate visualizations for high-danger and low-danger trajectories
    
    Args:
        analysis: Analysis results dict
        outputs_dir: Base directory containing trajectory files or trajectory folders
        trajectory_numbers: Optional list of trajectory numbers (for searching in subdirectories)
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create visualization directory in base outputs
    base_outputs = Path(__file__).parent / "outputs"
    viz_dir = base_outputs / 'danger_visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"Creating visualizations in: {viz_dir}\n")
    
    def find_trajectory_file(filename: str) -> Path:
        """Find trajectory file, searching in subdirectories if needed"""
        # First try direct path
        direct_path = outputs_dir / filename
        if direct_path.exists():
            return direct_path
        
        # If trajectory_numbers specified, search in those subdirectories
        if trajectory_numbers:
            for traj_num in trajectory_numbers:
                subdir_path = outputs_dir / str(traj_num) / filename
                if subdir_path.exists():
                    return subdir_path
        
        # Fallback: search all subdirectories
        for subdir in outputs_dir.iterdir():
            if subdir.is_dir():
                subdir_path = subdir / filename
                if subdir_path.exists():
                    return subdir_path
        
        # Return direct path even if not found (will error later)
        return direct_path
    
    # Visualize highest danger trajectories
    print("Generating HIGH DANGER trajectory visualizations...")
    for rank, (map_num, filename, danger_score) in enumerate(analysis['highest_danger'], 1):
        traj_path = find_trajectory_file(filename)
        output_name = f"high_danger_{rank}_map_{map_num}_danger_{danger_score:.3f}.png"
        output_path = viz_dir / output_name
        
        print(f"  [{rank}/5] Map {map_num} (danger: {danger_score:.4f})", end='')
        try:
            trajectory_viz.create_static_visualization(
                str(traj_path),
                output_path=str(output_path),
                sample_rate=10
            )
            print(f" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
    
    # Visualize lowest danger trajectories
    print("\nGenerating LOW DANGER trajectory visualizations...")
    for rank, (map_num, filename, danger_score) in enumerate(analysis['lowest_danger'], 1):
        traj_path = find_trajectory_file(filename)
        output_name = f"low_danger_{rank}_map_{map_num}_danger_{danger_score:.3f}.png"
        output_path = viz_dir / output_name
        
        print(f"  [{rank}/5] Map {map_num} (danger: {danger_score:.4f})", end='')
        try:
            trajectory_viz.create_static_visualization(
                str(traj_path),
                output_path=str(output_path),
                sample_rate=10
            )
            print(f" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Visualizations saved to: {viz_dir}")
    print("=" * 60)


def analyze_all_trajectories_multiple_models(
    outputs_dir: Path = None,
    trajectory_numbers: List[int] = None
) -> Dict:
    """
    Analyze all trajectories using multiple value models sequentially.
    Each model uses its corresponding motor response and obs noise by index.
    
    Args:
        outputs_dir: Base directory containing trajectory JSON files or trajectory folders
        trajectory_numbers: Optional list of trajectory numbers to analyze (collects from multiple folders)
    
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
    print(f"ANALYZING TRAJECTORIES WITH {num_models} VALUE MODELS")
    print(f"{'='*70}\n")
    
    all_model_results = []
    
    # Run each model sequentially
    for model_idx, model_name in enumerate(VALUE_MODELS):
        print(f"\n{'='*70}")
        print(f"MODEL {model_idx + 1}/{num_models}: {model_name}")
        print(f"{'='*70}")
        
        # Get corresponding motor response and obs noise (for consistency, though not used in danger analysis)
        motor_response = MOTOR_RESPONSES[model_idx]
        obs_noise_std = OBS_NOISE_STDS[model_idx]
        
        # Analyze trajectories with this model
        model_results = analyze_all_trajectories(
            outputs_dir=outputs_dir,
            trajectory_numbers=trajectory_numbers,
            model=model_name
        )
        
        if model_results:
            # Convert to consistent format matching 04_score_difficulty.py structure
            model_result_dict = {
                'model': model_results['model'],
                'motor_response': motor_response,
                'obs_noise_std': obs_noise_std,
                'total_maps': len(model_results['results']),
                'all_results': [
                    {
                        'map_number': int(m),
                        'filename': f,
                        'danger_score': float(r)
                    }
                    for m, f, r in model_results['results']
                ],
                'highest_danger': [
                    {'map_number': int(m), 'filename': f, 'danger_score': float(r)}
                    for m, f, r in model_results['highest_danger']
                ],
                'lowest_danger': [
                    {'map_number': int(m), 'filename': f, 'danger_score': float(r)}
                    for m, f, r in model_results['lowest_danger']
                ],
                'stats': {k: float(v) for k, v in model_results['stats'].items()}
            }
            all_model_results.append(model_result_dict)
    
    # Prepare combined output
    combined_results = {
        'model_results': all_model_results
    }
    
    # Save combined results
    # If multiple trajectory numbers, save to base outputs directory
    if trajectory_numbers and len(trajectory_numbers) > 1:
        save_dir = Path(__file__).parent / "outputs"
        output_file = save_dir / "danger_analysis_results.json"
    else:
        output_file = outputs_dir / "danger_analysis_results.json"
    
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
        if model_result['stats']:
            print(f"  Mean danger score: {model_result['stats']['mean']:.4f}")
        print()
    
    return combined_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze danger of Flappy Bird trajectories")
    parser.add_argument(
        '--trajectory-number', '-t',
        type=int,
        nargs='+',
        default=None,
        help='Trajectory number(s) to analyze (can specify multiple, e.g., -t 441 12 352)'
    )
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default=None,
        help='Custom outputs directory path'
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
        print(f"DANGER ANALYSIS INPUT: {output_dir}")
        print(f"{'='*70}")

        # Run analysis - use multi-model by default
        if args.single_model:
            # Single model mode (original behavior)
            analysis = analyze_all_trajectories(outputs_dir=output_dir, trajectory_numbers=trajectory_numbers)
            if analysis:
                print_results(analysis)
                
                # Save results to file
                # If multiple trajectory numbers, save to base outputs directory
                if trajectory_numbers and len(trajectory_numbers) > 1:
                    save_dir = Path(__file__).parent / "outputs"
                    output_file = save_dir / 'danger_analysis_results.json'
                else:
                    output_file = output_dir / 'danger_analysis_results.json'
                with open(output_file, 'w') as f:
                    # Convert to JSON-serializable format
                    # Include all_results with consistent field names (map_number instead of map)
                    json_data = {
                        'model': analysis['model'],
                        'total_maps': len(analysis['results']),
                        'all_results': [
                            {
                                'map_number': int(m),
                                'filename': f,
                                'danger_score': float(r)
                            }
                            for m, f, r in analysis['results']
                        ],
                        'highest_danger': [
                            {'map_number': int(m), 'filename': f, 'danger_score': float(r)}
                            for m, f, r in analysis['highest_danger']
                        ],
                        'lowest_danger': [
                            {'map_number': int(m), 'filename': f, 'danger_score': float(r)}
                            for m, f, r in analysis['lowest_danger']
                        ],
                        'stats': {k: float(v) for k, v in analysis['stats'].items()}
                    }
                    json.dump(json_data, f, indent=2)
                print(f"\nResults saved to: {output_file}")
                
                # Generate visualizations for high and low danger trajectories
                visualize_danger_trajectories(analysis, output_dir, trajectory_numbers=trajectory_numbers)
        else:
            # Multi-model mode (default)
            analyze_all_trajectories_multiple_models(
                outputs_dir=output_dir,
                trajectory_numbers=trajectory_numbers
            )
            
            # Note: Visualizations are skipped in multi-model mode since we have multiple models
            # You could extend this to visualize for each model if needed


if __name__ == "__main__":
    main()
