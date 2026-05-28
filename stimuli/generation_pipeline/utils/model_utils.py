from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch


def safe_torch_load(path: str, map_location: str | torch.device = "cpu", weights_only: bool = False) -> Dict[str, Any]:
    """Load a checkpoint robustly, always returning a dict-like object when possible.

    If the file contains a raw state_dict (tensor dict), return it as-is. Otherwise,
    expect a dict with at least 'model_state_dict'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    # Normalize to dict for downstream consumers
    if isinstance(checkpoint, dict):
        return checkpoint
    # Some very old saves may have only a raw state dict
    if isinstance(checkpoint, torch.nn.modules.module.Module):
        # Rare case: a whole module was saved; extract its state_dict
        return {"model_state_dict": checkpoint.state_dict()}
    if isinstance(checkpoint, (list, tuple)):
        return {"data": checkpoint}
    # Assume raw state dict
    return {"model_state_dict": checkpoint}


def extract_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract the model state dict from a flexible checkpoint format.

    Supports multiple conventions:
    - 'model_state_dict' (PPO in agents/)
    - 'network_state_dict' (PPO in follow/)
    - raw dict treated as state dict (very old saves)
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "network_state_dict" in checkpoint:
            return checkpoint["network_state_dict"]
        # Some saves may put tensors at the top-level
        # Detect by checking for any tensor values
        any_tensor_values = any(hasattr(v, 'shape') for v in checkpoint.values())
        if any_tensor_values:
            return checkpoint
    # Fallback empty dict
    return {}


def _is_ppo_checkpoint(checkpoint: Dict[str, Any], model_state_dict: Dict[str, Any]) -> bool:
    try:
        if isinstance(model_state_dict, dict):
            has_fc1 = "fc1.weight" in model_state_dict  # PPO in agents/
            has_actor = any(k.startswith("actor.") for k in model_state_dict.keys())
            has_critic = any(k.startswith("critic.") for k in model_state_dict.keys())
            # PPO in follow/ uses actor_head/critic_head/shared_layers naming
            has_follow_heads = any(k.startswith("actor_head.") for k in model_state_dict.keys()) and any(
                k.startswith("critic_head.") for k in model_state_dict.keys()
            )
            has_shared0 = any(k.endswith("shared_layers.0.weight") or k == "shared_layers.0.weight" for k in model_state_dict.keys())
            if has_fc1 or (has_actor and has_critic) or has_follow_heads or has_shared0:
                return True
        if isinstance(checkpoint, dict):
            hp = checkpoint.get("hparams", {})
            if isinstance(hp, dict) and ("clip_epsilon" in hp or "gae_lambda" in hp):
                return True
    except Exception:
        pass
    return False


def detect_agent_type_from_checkpoint(checkpoint: Dict[str, Any]) -> str:
    """Detect supported agent type from a loaded checkpoint."""
    msd = extract_model_state_dict(checkpoint)
    if _is_ppo_checkpoint(checkpoint, msd):
        return "ppo"
    raise ValueError("Unsupported checkpoint type: only PPO checkpoints are supported.")


def detect_state_size_from_msd(model_state_dict: Dict[str, Any]) -> int:
    """Infer PPO input state size from known first-layer keys."""
    if not isinstance(model_state_dict, dict):
        return 5
    if "fc1.weight" in model_state_dict:  # PPO ActorCritic in agents/
        return int(model_state_dict["fc1.weight"].shape[1])
    # PPO in follow/ PPONetwork shared_layers.0.weight
    if "shared_layers.0.weight" in model_state_dict:
        return int(model_state_dict["shared_layers.0.weight"].shape[1])
    return 5


def get_model_config(model_path: str) -> Dict[str, Any]:
    """Extract model configuration from a saved file.

    Returns a dict with keys:
    - model_type: 'ppo'
    - state_size: int (3 or 5 typically)
    - simple: False, retained for compatibility with existing metadata fields
    """
    checkpoint = safe_torch_load(model_path, map_location="cpu", weights_only=False)
    msd = extract_model_state_dict(checkpoint)
    model_type = detect_agent_type_from_checkpoint(checkpoint)
    state_size = detect_state_size_from_msd(msd)
    if model_type == "ppo":
        return {"model_type": "ppo", "state_size": state_size, "simple": False}
    raise ValueError(f"Unsupported model type: {model_type}")


def instantiate_agent(model_path: str, device: str = "cpu", verbose: bool = False):
    """Instantiate a PPO agent from a model file and load weights.

    Returns a tuple (agent, config_dict).
    """
    # Lazy import to avoid circular deps when this util is used by recording tools.
    from agents.ppo_agent import PPO_agent  # type: ignore

    # Resolve model path if a bare filename or non-existent path was provided
    resolved_path = model_path
    try:
        if not os.path.isabs(resolved_path) or not os.path.exists(resolved_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(project_root)
            candidate_models = os.path.join(project_root, 'models', os.path.basename(model_path))
            candidate_follow = os.path.join(project_root, 'follow', 'models', os.path.basename(model_path))
            if os.path.exists(candidate_models):
                resolved_path = candidate_models
            elif os.path.exists(candidate_follow):
                resolved_path = candidate_follow
    except Exception:
        pass

    cfg = get_model_config(resolved_path)
    model_type = cfg.get("model_type", "ppo")
    state_size = int(cfg.get("state_size", 5))
    if model_type == "ppo":
        # Try standard PPO agent (agents/ppo_agent)
        try:
            agent = PPO_agent(device=device, state_size=state_size, verbose=verbose)
            loaded = agent.load_model(resolved_path)
            # Some loaders return bool; if False, attempt follow PPO next
            if loaded is False:
                raise RuntimeError("agents.PPO_agent refused checkpoint (likely follow PPO)")
            agent.model.eval()
            return agent, cfg
        except Exception:
            # Fallback to follow PPO agent and wrap with a minimal eval adapter
            try:
                from follow.follow_agents.ppo_agent import PPOAgent as FollowPPOAgent  # type: ignore
                follow_agent = FollowPPOAgent(state_size=state_size, device=device)
                follow_agent.load_model(resolved_path)

                class FollowPPOEvalAdapter:
                    def __init__(self, core):
                        self.core = core
                        try:
                            self.state_size = int(core.get_input_state_size())
                        except Exception:
                            self.state_size = int(getattr(core, 'state_size', state_size))
                        self.device = getattr(core, 'device', device)

                    def get_q_values(self, state):
                        import torch
                        s = align_state_to_size(state, self.state_size)
                        with torch.no_grad():
                            st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                            logits, _value = self.core.network(st)
                            return logits.squeeze(0).detach().cpu().numpy()

                adapter = FollowPPOEvalAdapter(follow_agent)
                return adapter, {**cfg, "model_type": "ppo", "state_size": adapter.state_size}
            except Exception as e:
                # Re-raise the original error context for visibility
                raise e
    raise ValueError(f"Unsupported model type: {model_type}")


def align_state_to_size(state: Any, expected_size: int) -> list:
    """Truncate or pad a state sequence to match expected_size.

    - If state is longer, truncate from the end
    - If shorter, pad with zeros
    - Always returns a plain Python list of length expected_size
    """
    try:
        seq = list(state)
    except Exception:
        return [0.0] * expected_size
    if len(seq) >= expected_size:
        return seq[:expected_size]
    return seq + [0.0] * (expected_size - len(seq))


def apply_ppo_hparams(ppo_agent_obj: Any, hyperparameter: Dict[str, Any]) -> None:
    """Apply PPO-specific hyperparameters to a PPO_agent instance.

    This centralizes mapping from grid keys to agent internals.
    """
    # Static hparams
    if hasattr(ppo_agent_obj, 'hparams'):
        hp = ppo_agent_obj.hparams
        # Epochs (allow aliases)
        if 'ppo_epochs' in hyperparameter:
            hp.ppo_epochs = int(hyperparameter['ppo_epochs'])
        if 'n_epochs' in hyperparameter:
            hp.ppo_epochs = int(hyperparameter['n_epochs'])
        if 'minibatch_size' in hyperparameter:
            hp.minibatch_size = int(hyperparameter['minibatch_size'])
        if 'entropy_coef' in hyperparameter:
            hp.entropy_coef = float(hyperparameter['entropy_coef'])
        if 'value_coef' in hyperparameter:
            hp.value_coef = float(hyperparameter['value_coef'])
        if 'gae_lambda' in hyperparameter:
            hp.gae_lambda = float(hyperparameter['gae_lambda'])
        if 'max_grad_norm' in hyperparameter:
            hp.max_grad_norm = float(hyperparameter['max_grad_norm'])
    # Value clipping
    if 'value_clip_epsilon' in hyperparameter and hasattr(ppo_agent_obj, 'value_clip_epsilon'):
        ppo_agent_obj.value_clip_epsilon = float(hyperparameter['value_clip_epsilon'])
        if hasattr(ppo_agent_obj, 'use_value_clipping'):
            ppo_agent_obj.use_value_clipping = True
    # Dropout handled via p_dropout property by Game already; lr, rollout sizing, gamma are set by Game's PPO loop.
    # Allow optional aliases that are consumed by Game (not the agent):
    # - 'rollout_length' / 'steps_per_update'
    # - 'total_timesteps'
    # These are not applied to the agent directly.


