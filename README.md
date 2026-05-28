# Enjoyable Action Sequences

This repository contains the stimulus-generation code, browser experiment, and analysis notebooks for our study on what makes action sequences enjoyable to watch. The project uses short Flappy Bird-style gameplay videos as controlled stimuli and asks participants to judge how enjoyable, dangerous, or difficult those trajectories appear.

At a high level, the experiment tests how enjoyment relates to two computationally defined properties of a trajectory:

- `dangerousness`: how close the observed path comes to failure or risky states.
- `difficulty`: how hard the same map is for trained agents to complete.

The study separates properties of the player's movement from properties of the environment by generating many map variants around selected seed trajectories, scoring those variants, and then selecting videos that span low/high dangerousness and low/high difficulty conditions.

## Method Overview

The workflow has three main parts:

1. Create stimulus sets by selecting seed trajectories, generating map variants, scoring each map for dangerousness and difficulty, and rendering selected trajectories as videos.
2. Run a web-based watch-and-rate experiment in which participants view short gameplay videos, provide slider ratings, make pairwise comparisons, and complete an exit survey.
3. Analyze the data (perceived enjoyment, dangerousness, and difficulty) and relate those judgments to the computational stimulus scores reported in the paper.

## Repo

- `stimuli/`: stimulus construction, upload notebooks, and the Flappy Bird stimulus-generation pipeline.
- `stimuli/generation_pipeline/`: scripts for recording trajectories, generating maps, scoring dangerousness and difficulty, selecting stimuli, and rendering videos.
- `experiment/`: jsPsych browser experiment plus Node/Socket.IO services for stimulus assignment and data writes.
- `analysis/`: notebooks for cleaning behavioral data, estimating ratings and pairwise preferences, and relating human judgments to computed scores.
- `data/`: anonymized analysis-ready CSV files used by the notebooks.
- `environment.yml`: Python environment for stimulus generation and analysis.

## Stimulus Generation

The stimulus pipeline is documented in `stimuli/generation_pipeline/README.md`. In brief, it records trajectories from trained agents, selects high- and low-amplitude seeds, generates map variants, scores difficulty and dangerousness, selects extreme stimuli, and renders videos for the experiment.

Create the Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate flappy
```

Then follow the numbered scripts in `stimuli/generation_pipeline/`.

## Web Experiment

The experiment code is in `experiment/`. It serves a jsPsych task through `app.js` and uses `store.js` to read stimulus assignments from MongoDB and write participant data back to MongoDB.

See `experiment/README.md` for setup details.

## Analysis

The main analysis notebook is `analysis/analysis.ipynb`. It uses the rating and pairwise-comparison data in `data/` to summarize participant judgments and compare human ratings of enjoyment, dangerousness, and difficulty against the computational stimulus scores. All analyses covered in the paper can be found in this notebook.