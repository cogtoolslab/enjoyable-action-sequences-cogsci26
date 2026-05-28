# Stimuli Selection Pipeline

This folder contains the code used to generate, score, select, and render Flappy Bird stimulus videos.

The workflow is:

1. Record many seed trajectories from a trained agent.
2. Inspect those trajectories and choose representative high- and low-amplitude seeds.
3. Generate many map variants from each selected seed.
4. Score each map for difficulty and danger.
5. Select extreme danger/difficulty stimuli.
6. Render the selected trajectories as videos.

PPO models can be trained with `models/main_ppo.py` and then used by the generation scripts below.

## Setup

First, create the `flappy` conda environment from the repository root:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate flappy
```

## Folder Layout

```text
stimuli/generation_pipeline/
  01_record_seed_trajectories.py      record a large pool of agent trajectories
  02_select_seed_trajectories.ipynb   inspect trajectories and choose seed examples
  03_create_maps.py                   generate map variants from selected seeds
  04_score_difficulty.py              estimate map difficulty
  05_score_danger.py                    estimate map danger
  06_select_extreme_stimuli.py        find low/high danger and difficulty stimuli
  07_render_visuals.py                 render trajectory JSON files as MP4s/PNGs
  core/                               game loop, objects, and constants
  recording/                          recording and rendering helpers
  agents/                             trained agent implementations
  models/                             trained model checkpoints
  assets/                             images used by the game and videos
  initial_trajectories/               selected seed trajectories
  outputs/                            generated maps, analyses, images, and videos
```

## Quick Example Test

This runs a tiny version of the default pipeline using the selected seed files already present in `initial_trajectories/`.

```bash
python stimuli/generation_pipeline/03_create_maps.py \
  --num-maps 2

python stimuli/generation_pipeline/04_score_difficulty.py \
  --num-trials 2 \
  --parallel

python stimuli/generation_pipeline/05_score_danger.py

python stimuli/generation_pipeline/06_select_extreme_stimuli.py \
  --max-attempts 1 \
  --num-trials 1 \
  --num-workers 1
```

## Full Workflow

### 1. Record Seed Trajectories

```bash
python stimuli/generation_pipeline/01_record_seed_trajectories.py
```

This records many trajectories from the configured trained models and writes them to model-specific folders:

```text
initial_trajectories/unlimited/
initial_trajectories/mr_2_1_10/
initial_trajectories/mr_5_2_20/
```

### 2. Select Representative Seeds

Open:

```text
stimuli/generation_pipeline/02_select_seed_trajectories.ipynb
```

Use this notebook to inspect the generated trajectories and choose representative high- and low-amplitude examples. Run it for each model you want in the default six-seed workflow.

With the last cell, the selected seed files must be copied into `initial_trajectories/` with these names:

```text
initial_trajectories/mr_2_1_10_trajectory_high_amplitude.json
initial_trajectories/mr_2_1_10_trajectory_low_amplitude.json
initial_trajectories/mr_5_2_20_trajectory_high_amplitude.json
initial_trajectories/mr_5_2_20_trajectory_low_amplitude.json
initial_trajectories/unlimited_trajectory_high_amplitude.json
initial_trajectories/unlimited_trajectory_low_amplitude.json
```

### 3. Generate Map Variants

Default run for all six selected seeds:

```bash
python stimuli/generation_pipeline/03_create_maps.py
```

This creates one output folder per seed:

```text
outputs/mr_2_1_10_high_amplitude/
outputs/mr_2_1_10_low_amplitude/
outputs/mr_5_2_20_high_amplitude/
outputs/mr_5_2_20_low_amplitude/
outputs/unlimited_high_amplitude/
outputs/unlimited_low_amplitude/
```

To change the number of generated maps per seed:

```bash
python stimuli/generation_pipeline/03_create_maps.py \
  --num-maps 1000
```

### 4. Score Difficulty

Default run for all six output folders:

```bash
python stimuli/generation_pipeline/04_score_difficulty.py \
  --num-trials 100 \
  --parallel
```

This writes `map_difficulty_results.json` inside each output folder.

To score one folder only:

```bash
python stimuli/generation_pipeline/04_score_difficulty.py \
  --outputs-dir unlimited_low_amplitude \
  --num-trials 100 \
  --parallel
```

### 5. Score Danger

Default run for all six output folders:

```bash
python stimuli/generation_pipeline/05_score_danger.py
```

This writes `danger_analysis_results.json` inside each output folder.

To score one folder only:

```bash
python stimuli/generation_pipeline/05_score_danger.py \
  --outputs-dir unlimited_low_amplitude
```

### 6. Select Extreme Stimuli

Default run for all six selected seeds:

```bash
python stimuli/generation_pipeline/06_select_extreme_stimuli.py \
  --max-attempts 10000 \
  --num-trials 100
```

For each seed, this script expects the matching output folder from steps 3-5. For example:

```text
initial_trajectories/unlimited_trajectory_low_amplitude.json
outputs/unlimited_low_amplitude/map_difficulty_results.json
outputs/unlimited_low_amplitude/danger_analysis_results.json
```

Selected stimulus JSON files, PNG visualizations, and MP4 videos are written to `outputs/stimuli_trajectories/`.

## If Training Your Own PPO Models

To train or evaluate a PPO model, edit the configuration variables at the top of `stimuli/generation_pipeline/models/main_ppo.py`, then run:

```bash
python stimuli/generation_pipeline/models/main_ppo.py
```

Common variables to edit:

```python
MODE = "train"  # or "eval"
MODEL_NAME = "value_ppo_custom.pth"
RESUME_MODEL = None
EVAL_MODEL = MODEL_NAME

UPDATES = 100
ROLLOUT_LENGTH = 2048

MOTOR_MEAN = None
MOTOR_STD = None
OBS_NOISE_STD = 0.0
```

For a motor-response / observation-noise model, use values like:

```python
MODEL_NAME = "value_ppo_mr_2_1_on_10.pth"
MOTOR_MEAN = 2
MOTOR_STD = 1
OBS_NOISE_STD = 10
```

To evaluate, set:

```python
MODE = "eval"
EVAL_MODEL = "value_ppo_custom.pth"
EVAL_RUNS_ONLY = 50
```

Saved checkpoints are written to `stimuli/generation_pipeline/models/` unless `OUTPUT_PATH` is set to an explicit path.

## Standalone Video Rendering

Use `07_render_visuals.py` if you already have trajectory JSON files and only want to render videos or visualizations:

```bash
python stimuli/generation_pipeline/07_render_visuals.py video
python stimuli/generation_pipeline/07_render_visuals.py visualize trajectory.json --mode static --output trajectory.png
```

## Notes

- Large runs can take a long time because difficulty scoring evaluates several models over many trials.

