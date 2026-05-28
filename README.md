# What makes an action sequence enjoyable to watch?

This work will be presented at CogSci 2026. See the [project page](https://cogtoolslab.github.io/enjoyable-action-sequences-cogsci26/) for an overview.

This repository contains the stimulus-generation code, experiment, and analysis notebooks for our study on what makes action sequences enjoyable to watch. The project uses short Flappy Bird-style gameplay videos as controlled stimuli and asks participants to judge how enjoyable, dangerous, or difficult those trajectories appear.

At a high level, the experiment tests how enjoyment relates to two computationally defined properties of a trajectory:

- `dangerousness`: how close the observed path comes to failure or risky states.
- `map difficulty`: how hard a map is for agents to complete.

## Repo Structure

- `stimuli/`: stimulus construction, upload notebooks, and the Flappy Bird stimulus-generation pipeline.
- `stimuli/generation_pipeline/`: scripts for recording trajectories, generating maps, scoring dangerousness and difficulty, selecting stimuli, and rendering videos.
- `experiment/`: jsPsych browser experiment plus Node/Socket.IO services for stimulus assignment and data writes.
- `analysis/`: notebooks for cleaning behavioral data, estimating ratings and pairwise preferences, and relating human judgments to computed scores.
- `data/`: anonymized analysis-ready CSV files used by the notebooks.
- `environment.yml`: Python environment for stimulus generation and analysis.