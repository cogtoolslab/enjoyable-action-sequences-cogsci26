"""Train and evaluate PPO models in the local Flappy Bird environment.

Edit the variables in the configuration section below, then run:

    python stimuli/generation_pipeline/models/main_ppo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PIPELINE_ROOT / "models"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from core.game import Game  # noqa: E402
from utils.model_utils import apply_ppo_hparams  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set MODE to "train" or "eval".
MODE = "train"

# Model paths. Relative names are resolved inside stimuli/generation_pipeline/models/.
MODEL_NAME = "value_ppo_example.pth"
OUTPUT_PATH = str(MODELS_DIR / MODEL_NAME)
RESUME_MODEL = None  # Optional checkpoint for continued training.
EVAL_MODEL = MODEL_NAME

# Environment settings.
DEVICE = "cpu"  # "cpu", "cuda", or "mps"
STATE_SIZE = 5  # 3 or 5; PPO internally receives one extra pending-jump flag.
MAX_SCORE = 200
SEED = None
VERBOSE = False

# Limitations. Set both motor values to None to disable motor-response delay.
MOTOR_MEAN = 2
MOTOR_STD = 1
OBS_NOISE_STD = 10.0

# Training settings.
UPDATES = 100
ROLLOUT_LENGTH = 2048
EVAL_EVERY = 10
EVAL_RUNS = 20
CHECKPOINT_EVERY = 0  # 0 disables periodic checkpoints.
SAVE_BEST = False  # If True, save only when evaluation mean improves.

# PPO hyperparameters.
LR = 3e-4
GAMMA = 0.9
MINIBATCH_SIZE = 128
PPO_EPOCHS = 8
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5
VALUE_CLIP_EPSILON = 0.2

# Evaluation settings.
EVAL_RUNS_ONLY = 50
DRAW_EVAL = False


class Config:
    """Runtime config built from the editable variables above."""

    def __init__(self, **overrides):
        self.mode = MODE
        self.model_name = MODEL_NAME
        self.output_path = OUTPUT_PATH
        self.resume_model = RESUME_MODEL
        self.eval_model = EVAL_MODEL
        self.device = DEVICE
        self.state_size = STATE_SIZE
        self.max_score = MAX_SCORE
        self.seed = SEED
        self.verbose = VERBOSE
        self.motor_mean = MOTOR_MEAN
        self.motor_std = MOTOR_STD
        self.obs_noise_std = OBS_NOISE_STD
        self.updates = UPDATES
        self.rollout_length = ROLLOUT_LENGTH
        self.eval_every = EVAL_EVERY
        self.eval_runs = EVAL_RUNS
        self.checkpoint_every = CHECKPOINT_EVERY
        self.save_best = SAVE_BEST
        self.lr = LR
        self.gamma = GAMMA
        self.minibatch_size = MINIBATCH_SIZE
        self.ppo_epochs = PPO_EPOCHS
        self.entropy_coef = ENTROPY_COEF
        self.value_coef = VALUE_COEF
        self.gae_lambda = GAE_LAMBDA
        self.max_grad_norm = MAX_GRAD_NORM
        self.value_clip_epsilon = VALUE_CLIP_EPSILON
        self.eval_runs_only = EVAL_RUNS_ONLY
        self.draw_eval = DRAW_EVAL

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(self, key, value)


def resolve_model_path(path_or_name: str | None) -> str | None:
    """Resolve a model path from either an absolute/relative path or a models/ filename."""
    if not path_or_name:
        return None

    path = Path(path_or_name)
    if path.is_absolute():
        return str(path)

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return str(cwd_candidate)

    return str(MODELS_DIR / path.name)


def make_env(config: Config, model_path: str | None = None, verbose: bool = False) -> Game:
    """Create a PPO Flappy Bird environment with the requested limitations."""
    motor_response = None
    if config.motor_mean is not None and config.motor_std is not None:
        motor_response = (config.motor_mean, config.motor_std)

    return Game(
        "ppo_agent",
        config.device,
        model_path=model_path,
        action_fail_probability=0.0,
        sticky_keys=0,
        state_size=config.state_size,
        verbose=verbose,
        motor_response=motor_response,
        obs_noise_std=config.obs_noise_std,
    )


def configure_agent(env: Game, config: Config) -> None:
    """Apply PPO hyperparameters to the environment's agent."""
    env.agent.lr = config.lr
    env.agent.gamma = config.gamma
    apply_ppo_hparams(
        env.agent,
        {
            "minibatch_size": config.minibatch_size,
            "ppo_epochs": config.ppo_epochs,
            "entropy_coef": config.entropy_coef,
            "value_coef": config.value_coef,
            "gae_lambda": config.gae_lambda,
            "max_grad_norm": config.max_grad_norm,
            "value_clip_epsilon": config.value_clip_epsilon,
        },
    )


def collect_rollout(env: Game, rollout_length: int, max_score: int) -> list[int]:
    """Collect at least rollout_length PPO transitions by running full episodes."""
    env.train = True
    scores = []
    while len(env.agent.buffer[0]) < rollout_length:
        score = env.main(draw=False, max_score=max_score)
        scores.append(int(score))
    env.train = False
    return scores


def evaluate(env: Game, runs: int, max_score: int, draw: bool = False) -> np.ndarray:
    """Evaluate the current policy greedily."""
    original_train = env.train
    env.train = False
    scores = []
    for _ in range(runs):
        scores.append(env.main(draw=draw, max_score=max_score))
    env.train = original_train
    return np.asarray(scores, dtype=float)


def train(config: Config) -> None:
    """Train a PPO model and save checkpoints under models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_model_path(config.resume_model)
    save_path = Path(resolve_model_path(config.output_path or config.model_name) or (MODELS_DIR / config.model_name))

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env = make_env(config, model_path=resume_path, verbose=config.verbose)
    configure_agent(env, config)

    best_mean = -float("inf")
    print("Training PPO agent")
    print(f"  output: {save_path}")
    print(f"  updates: {config.updates}, rollout_length: {config.rollout_length}")
    print(f"  motor_response: {None if config.motor_mean is None else (config.motor_mean, config.motor_std)}")
    print(f"  obs_noise_std: {config.obs_noise_std}")
    if resume_path:
        print(f"  resume: {resume_path}")

    for update_idx in range(1, config.updates + 1):
        scores = collect_rollout(env, config.rollout_length, config.max_score)
        loss = env.agent.train()

        if update_idx % config.eval_every == 0 or update_idx == 1 or update_idx == config.updates:
            eval_scores = evaluate(env, config.eval_runs, config.max_score)
            mean_score = float(eval_scores.mean()) if len(eval_scores) else 0.0
            train_mean = float(np.mean(scores)) if scores else 0.0
            info = getattr(env.agent, "last_train_info", {})
            print(
                f"update {update_idx:4d}/{config.updates} | "
                f"train_mean={train_mean:6.2f} | eval_mean={mean_score:6.2f} | "
                f"loss={loss:9.4f} | kl={info.get('approx_kl', 0.0):.4f} | "
                f"ev={info.get('explained_var', 0.0):.3f}"
            )

            if config.save_best and mean_score >= best_mean:
                best_mean = mean_score
                env.agent.save_model(str(save_path))
                print(f"  saved new best model to {save_path}")

        if config.checkpoint_every and update_idx % config.checkpoint_every == 0:
            checkpoint_path = save_path.with_name(f"{save_path.stem}_update_{update_idx}{save_path.suffix}")
            env.agent.save_model(str(checkpoint_path))
            print(f"  checkpoint saved to {checkpoint_path}")

    if not config.save_best:
        env.agent.save_model(str(save_path))
        print(f"Saved final model to {save_path}")


def eval_model(config: Config) -> None:
    """Evaluate a PPO checkpoint."""
    model_path = resolve_model_path(config.eval_model)
    env = make_env(config, model_path=model_path, verbose=config.verbose)
    scores = evaluate(env, config.eval_runs_only, config.max_score, draw=config.draw_eval)
    print(f"Evaluated {model_path}")
    print(f"  runs: {config.eval_runs_only}")
    print(f"  mean: {scores.mean():.2f}")
    print(f"  std:  {scores.std():.2f}")
    print(f"  min:  {scores.min():.2f}")
    print(f"  max:  {scores.max():.2f}")


def main() -> None:
    config = Config()
    if config.mode == "train":
        train(config)
    elif config.mode == "eval":
        eval_model(config)
    else:
        raise ValueError("MODE must be 'train' or 'eval'.")


if __name__ == "__main__":
    main()
