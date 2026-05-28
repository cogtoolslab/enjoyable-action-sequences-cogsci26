# Enjoyable Action Sequences

This repository contains the stimulus-generation code, experiment, and analysis notebooks for our study on what makes action sequences enjoyable to watch. The project uses short Flappy Bird-style gameplay videos as controlled stimuli and asks participants to judge how enjoyable, dangerous, or difficult those trajectories appear.

At a high level, the experiment tests how enjoyment relates to two computationally defined properties of a trajectory:

- `dangerousness`: how close the observed path comes to failure or risky states.
- `difficulty`: how hard the same map is for trained agents to complete.

## Method Overview

The workflow has three main parts:

1. Create stimulus sets by selecting seed trajectories, generating map variants, scoring each map for dangerousness and difficulty, and rendering selected trajectories as videos.
2. Run a web-based watch-and-rate experiment in which participants view short gameplay videos, provide slider ratings, make pairwise comparisons, and complete an exit survey.
3. Analyze the data (perceived enjoyment, dangerousness, and difficulty) and relate those judgments to the computational stimulus scores reported in the paper.

## Repo Structure

- `stimuli/`: stimulus construction, upload notebooks, and the Flappy Bird stimulus-generation pipeline.
- `stimuli/generation_pipeline/`: scripts for recording trajectories, generating maps, scoring dangerousness and difficulty, selecting stimuli, and rendering videos.
- `experiment/`: jsPsych browser experiment plus Node/Socket.IO services for stimulus assignment and data writes.
- `analysis/`: notebooks for cleaning behavioral data, estimating ratings and pairwise preferences, and relating human judgments to computed scores.
- `data/`: anonymized analysis-ready CSV files used by the notebooks.
- `environment.yml`: Python environment for stimulus generation and analysis.