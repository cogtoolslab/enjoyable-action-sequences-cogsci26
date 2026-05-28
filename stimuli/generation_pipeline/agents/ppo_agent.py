import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from torch import nn
import torch.nn.functional as F
from utils.model_utils import safe_torch_load, extract_model_state_dict, detect_state_size_from_msd


@dataclass
class PPOHyperParams:
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float = 0.01


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, int] = (128, 128), p_dropout: float = 0.0):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.actor = nn.Linear(h2, 2)  # 2 actions: 0=do nothing, 1=jump
        self.critic = nn.Linear(h2, 1)
        self._p_dropout = float(p_dropout)
        self.dropout1 = nn.Dropout(self._p_dropout)
        self.dropout2 = nn.Dropout(self._p_dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


class PPO_agent:
    """
    Proximal Policy Optimization agent compatible with the existing Game loop.

    Contract expected by Game:
    - self.buffer is a list of 5 lists: [states, next_states, rewards, actions, terminal_flags]
    - act(state, train) returns an int action in {0,1}
    - train() consumes current buffer and performs PPO update
    - Attributes lr, epsilon, batch_size, gamma exist for compatibility with Game's scheduler
    """

    def __init__(self, device: str, state_size: int, verbose: bool = False, p_dropout: float = 0.0):
        self.device = device
        self.state_size = state_size
        self.verbose = verbose

        # Exposed attributes for compatibility with Game.train_agent
        self._lr: float = 3e-4
        self.batch_size: int = 2048  # trajectory target; Game controls update cadence
        self.gamma: float = 0.9
        self._epsilon_unused: float = 0.0  # kept for compatibility with older training code

        self.hparams = PPOHyperParams()
        self._p_dropout = float(p_dropout)
        self.use_value_clipping: bool = True
        self.value_clip_epsilon: float = 0.2

        # Networks
        self.model = ActorCritic(self.state_size, p_dropout=self._p_dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)

        # Experience buffer aligned with Game's expectations
        self.buffer: List[List] = [[], [], [], [], []]

        # Per-step storage aligned with buffer entries (old log-probs, state values)
        self.saved_log_probs: List[float] = []
        self.saved_values: List[float] = []

        self.training_steps: int = 0

    # Learning rate compat
    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float):
        self._lr = float(value)
        for g in self.optimizer.param_groups:
            g['lr'] = self._lr

    # Epsilon compat (unused by PPO)
    @property
    def epsilon(self) -> float:
        return self._epsilon_unused

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon_unused = float(value)

    @property
    def p_dropout(self) -> float:
        return self._p_dropout

    @p_dropout.setter
    def p_dropout(self, value: float):
        self._p_dropout = float(value)
        if hasattr(self.model, 'dropout1') and hasattr(self.model, 'dropout2'):
            self.model.dropout1.p = self._p_dropout
            self.model.dropout2.p = self._p_dropout

    # Action selection
    def act(self, state, train: bool) -> int:
        # Check if we're in trajectory collection mode (greedy but populate buffer)
        collecting_mode = getattr(self, '_collecting_trajectories', False)
        
        if train and not collecting_mode:
            self.model.train()
        else:
            self.model.eval()
            
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits, value = self.model(state_t)
        # Guard against invalid logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.zeros_like(logits)
            
        if train and not collecting_mode:
            # Standard training: stochastic actions
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)
            # Store scalars for later PPO update
            self.saved_log_probs.append(float(log_prob_t.detach().cpu().item()))
            self.saved_values.append(float(value.detach().cpu().item()))
            return int(action_t.item())
        elif collecting_mode:
            # Trajectory collection: greedy actions but populate buffer
            action = int(torch.argmax(logits).item())
            # Still populate saved values for buffer compatibility
            self.saved_log_probs.append(0.0)  # Dummy value
            self.saved_values.append(float(value.detach().cpu().item()))
            return action
        else:
            # Evaluation: greedy actions, no buffer
            return int(torch.argmax(logits).item())

    # GAE computation
    def _compute_gae(self, rewards, values, dones, gamma: float, lam: float, last_value: float):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_advantage = 0.0
        # Bootstrap with last_value when not terminal
        values_next = np.concatenate([values[1:], np.array([float(last_value)], dtype=np.float32)])
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values_next[t] * mask - values[t]
            last_advantage = delta + gamma * lam * mask * last_advantage
            advantages[t] = last_advantage
        returns = advantages + values
        return advantages, returns

    # PPO update
    def train(self) -> float:
        # Enable dropout during optimization
        self.model.train()
        num_samples = len(self.buffer[0])
        if num_samples == 0:
            return 0.0

        states = torch.tensor(self.buffer[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer[3], dtype=torch.int64, device=self.device)
        rewards_np = np.asarray(self.buffer[2], dtype=np.float32)
        dones_np = np.asarray(self.buffer[4], dtype=np.bool_)

        # Safety alignment: ensure saved arrays match transitions length
        if len(self.saved_log_probs) != num_samples or len(self.saved_values) != num_samples:
            min_len = min(num_samples, len(self.saved_log_probs), len(self.saved_values))
            states = states[:min_len]
            actions = actions[:min_len]
            rewards_np = rewards_np[:min_len]
            dones_np = dones_np[:min_len]
            self.saved_log_probs = self.saved_log_probs[:min_len]
            self.saved_values = self.saved_values[:min_len]
            num_samples = min_len

        old_log_probs = torch.tensor(self.saved_log_probs, dtype=torch.float32, device=self.device)
        values_np = np.asarray(self.saved_values, dtype=np.float32)

        # Compute bootstrap value for last next_state
        last_next_state = torch.tensor(self.buffer[1][-1], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _logits_tail, last_value_t = self.model(last_next_state)
        last_value = float(last_value_t.detach().cpu().item())

        adv_np, ret_np = self._compute_gae(
            rewards=rewards_np,
            values=values_np,
            dones=dones_np,
            gamma=self.gamma,
            lam=self.hparams.gae_lambda,
            last_value=last_value,
        )

        # Replace any non-finite numbers to avoid NaNs
        adv_np = np.nan_to_num(adv_np, nan=0.0, posinf=0.0, neginf=0.0)
        ret_np = np.nan_to_num(ret_np, nan=0.0, posinf=0.0, neginf=0.0)
        old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0, posinf=0.0, neginf=0.0)

        advantages = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret_np, dtype=torch.float32, device=self.device)
        # Robust normalization: avoid degrees-of-freedom warnings and division by zero
        if advantages.numel() > 1:
            std = advantages.std(unbiased=False)
            if not torch.isfinite(std) or std <= 0:
                std = torch.tensor(1.0, device=self.device, dtype=advantages.dtype)
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        else:
            # With a single sample, centering is fine; scaling is unstable
            advantages = advantages - advantages.mean()

        total_loss_value = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        total_updates = 0

        # Explained variance (before update) using current value predictions
        with torch.no_grad():
            _, values_before = self.model(states)
            var_y = torch.var(returns)
            if torch.isfinite(var_y) and var_y > 0:
                explained_var = 1.0 - torch.var(returns - values_before) / var_y
                explained_var = float(explained_var.detach().cpu().item())
            else:
                explained_var = 0.0
        indices = np.arange(num_samples)
        early_stop = False
        for _ in range(self.hparams.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.hparams.minibatch_size):
                end = start + self.hparams.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, value_pred = self.model(mb_states)
                # Guard against invalid logits in training
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.zeros_like(logits)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Stable ratio computation with clamping
                log_ratio = new_log_probs - mb_old_log_probs
                log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_value_clipping:
                    mb_values_old = torch.tensor(values_np[mb_idx], dtype=torch.float32, device=self.device)
                    value_pred_clipped = mb_values_old + torch.clamp(
                        value_pred - mb_values_old,
                        -self.value_clip_epsilon,
                        self.value_clip_epsilon,
                    )
                    value_loss_unclipped = (value_pred - mb_returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(value_pred, mb_returns)
                loss = policy_loss + self.hparams.value_coef * value_loss - self.hparams.entropy_coef * entropy
                if not torch.isfinite(loss):
                    # Skip unstable minibatch
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()

                # Logging/diagnostics
                with torch.no_grad():
                    # Approximate KL: mean(old_log - new_log)
                    approx_kl = torch.mean(mb_old_log_probs - new_log_probs).clamp_min(0).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.hparams.clip_epsilon).float()).item()

                total_loss_value += float(loss.detach().cpu().item())
                total_policy_loss += float(policy_loss.detach().cpu().item())
                total_value_loss += float(value_loss.detach().cpu().item())
                total_entropy += float(entropy.detach().cpu().item())
                total_clip_frac += clip_frac
                total_approx_kl += approx_kl
                total_updates += 1
                self.training_steps += 1

                # Early stop on excessive KL
                if approx_kl > 1.5 * self.hparams.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        # Aggregate last train info for logging
        if total_updates > 0:
            self.last_train_info = {
                'loss': total_loss_value / total_updates,
                'policy_loss': total_policy_loss / total_updates,
                'value_loss': total_value_loss / total_updates,
                'entropy': total_entropy / total_updates,
                'clip_frac': total_clip_frac / total_updates,
                'approx_kl': total_approx_kl / total_updates,
                'explained_var': explained_var,
                'updates': total_updates,
            }
        else:
            self.last_train_info = {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'clip_frac': 0.0,
                'approx_kl': 0.0,
                'explained_var': 0.0,
                'updates': 0,
            }

        # Clear buffers after update
        for i in range(5):
            self.buffer[i].clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()

        # Switch back to eval by default
        self.model.eval()
        return total_loss_value

    # Saving and loading utilities
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'p_dropout': self._p_dropout,
            'hparams': {
                'clip_epsilon': self.hparams.clip_epsilon,
                'entropy_coef': self.hparams.entropy_coef,
                'value_coef': self.hparams.value_coef,
                'ppo_epochs': self.hparams.ppo_epochs,
                'minibatch_size': self.hparams.minibatch_size,
                'gae_lambda': self.hparams.gae_lambda,
                'max_grad_norm': self.hparams.max_grad_norm,
            },
        }
        torch.save(save_dict, path)

    def load_model(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
        checkpoint = safe_torch_load(path, map_location=self.device, weights_only=False)
        msd = extract_model_state_dict(checkpoint)
        # Adapt network input size if needed
        try:
            inferred_input_size = detect_state_size_from_msd(msd)
            if inferred_input_size != self.state_size:
                if self.verbose:
                    print(f"⚠️  PPO model was trained with {inferred_input_size}-state input, adapting from current {self.state_size}.")
                self.state_size = inferred_input_size
                current_dropout = getattr(self, '_p_dropout', 0.0)
                self.model = ActorCritic(self.state_size, p_dropout=current_dropout).to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        except Exception:
            pass
        # Load weights
        self.model.load_state_dict(msd)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if isinstance(checkpoint, dict) and 'training_steps' in checkpoint:
            self.training_steps = int(checkpoint['training_steps'])
        if isinstance(checkpoint, dict) and 'p_dropout' in checkpoint:
            self.p_dropout = float(checkpoint['p_dropout'])  # use property to update modules
        if isinstance(checkpoint, dict) and 'hparams' in checkpoint and isinstance(checkpoint['hparams'], dict):
            hp = checkpoint['hparams']
            self.hparams.clip_epsilon = float(hp.get('clip_epsilon', self.hparams.clip_epsilon))
            self.hparams.entropy_coef = float(hp.get('entropy_coef', self.hparams.entropy_coef))
            self.hparams.value_coef = float(hp.get('value_coef', self.hparams.value_coef))
            self.hparams.ppo_epochs = int(hp.get('ppo_epochs', self.hparams.ppo_epochs))
            self.hparams.minibatch_size = int(hp.get('minibatch_size', self.hparams.minibatch_size))
            self.hparams.gae_lambda = float(hp.get('gae_lambda', self.hparams.gae_lambda))
            self.hparams.max_grad_norm = float(hp.get('max_grad_norm', self.hparams.max_grad_norm))
        return True

    # Inference utilities for compatibility with existing evaluation code
    def get_q_values(self, state, *args, **kwargs):
        """
        Return a 3-length array for analysis compatibility:
        [ V(s), logit_do_nothing, logit_jump ]

        - Uses the existing actor logits and critic value.
        - Ignores any extra positional/keyword args for backward compatibility.
        """
        self.model.eval()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value_t = self.model(state_t)
            v_s = float(value_t.detach().cpu().item())
            logits_np = logits.detach().cpu().numpy()
        import numpy as _np
        # Ensure shape (2,) for logits
        if logits_np.shape[-1] == 2:
            l0, l1 = float(logits_np[0]), float(logits_np[1])
        else:
            # Fallback: pad/trim
            l0 = float(logits_np[0]) if logits_np.size > 0 else 0.0
            l1 = float(logits_np[1]) if logits_np.size > 1 else 0.0
        return _np.array([v_s, l0, l1], dtype=float)

    # Phase 2: Value-only training utilities
    def freeze_policy(self):
        """Freeze actor parameters for value-only training."""
        for param in self.model.actor.parameters():
            param.requires_grad = False
        if self.verbose:
            print("Policy network frozen (actor parameters set to requires_grad=False)")

    def unfreeze_policy(self):
        """Unfreeze actor parameters."""
        for param in self.model.actor.parameters():
            param.requires_grad = True
        if self.verbose:
            print("Policy network unfrozen (actor parameters set to requires_grad=True)")

    def collect_trajectories(self, game_env, num_trajectories=100, verbose=True):
        """
        Collect trajectories from current policy for value training.
        Uses greedy (deterministic) actions from the optimal policy.
        
        Args:
            game_env: Game environment to run episodes in
            num_trajectories: Number of trajectories to collect
            verbose: Whether to print progress
            
        Returns:
            List of dicts with 'states' and 'returns' keys containing
            state-return pairs for value function training
        """
        trajectories = []
        
        if verbose:
            print(f"Collecting {num_trajectories} trajectories using greedy policy...")
        
        # Save original flags
        original_train_flag = game_env.train
        
        # Enable trajectory collection mode (greedy actions + buffer population)
        self._collecting_trajectories = True
        
        for traj_idx in range(num_trajectories):
            # Run episode in training mode to populate buffer
            # But agent will use greedy actions due to _collecting_trajectories flag
            game_env.train = True
            
            score = game_env.main(draw=False, track_rewards=True)
            
            # Get the rewards from the episode
            traj_rewards = []
            if hasattr(game_env, 'tracked_rewards') and game_env.tracked_rewards:
                traj_rewards = list(game_env.tracked_rewards)
            
            # Get the states from the buffer
            # Note: The buffer stores [states, next_states, rewards, actions, dones]
            traj_states = []
            if len(self.buffer[0]) > 0:
                traj_states = [list(state) if hasattr(state, '__iter__') else [state] for state in self.buffer[0]]
                
                # Also get rewards from buffer as backup
                if not traj_rewards and len(self.buffer[2]) > 0:
                    traj_rewards = list(self.buffer[2])
                
                # Clear buffer after extracting data
                for i in range(5):
                    self.buffer[i].clear()
                self.saved_log_probs.clear()
                self.saved_values.clear()
            
            # Compute Monte Carlo returns for this trajectory
            if len(traj_states) > 0 and len(traj_rewards) > 0:
                # Align lengths (sometimes there's a mismatch)
                min_len = min(len(traj_states), len(traj_rewards))
                traj_states = traj_states[:min_len]
                traj_rewards = traj_rewards[:min_len]
                
                mc_returns = []
                for i in range(len(traj_rewards)):
                    mc_return = 0.0
                    discount = 1.0
                    for j in range(i, len(traj_rewards)):
                        mc_return += discount * traj_rewards[j]
                        discount *= self.gamma
                    mc_returns.append(mc_return)
                
                trajectories.append({
                    'states': traj_states,
                    'returns': mc_returns,
                    'score': score
                })
            
            # Progress reporting
            if verbose and (traj_idx + 1) % 50 == 0:
                if len(trajectories) > 0:
                    recent_trajs = trajectories[-min(50, len(trajectories)):]
                    avg_score = np.mean([t['score'] for t in recent_trajs])
                    avg_return = np.mean([np.mean(t['returns']) for t in recent_trajs])
                    print(f"  Collected {traj_idx + 1}/{num_trajectories} trajectories | "
                          f"Avg Score (last 50): {avg_score:.1f} | "
                          f"Avg Return: {avg_return:.2f}")
        
        # Restore original flags
        game_env.train = original_train_flag
        self._collecting_trajectories = False
        
        if verbose:
            if len(trajectories) > 0:
                total_samples = sum(len(t['states']) for t in trajectories)
                avg_score = np.mean([t['score'] for t in trajectories])
                print(f"Collected {len(trajectories)} trajectories with {total_samples} state-return pairs")
                print(f"Average score: {avg_score:.1f}")
            else:
                print("WARNING: No trajectories collected! Buffer was empty.")
        
        return trajectories

    def train_value_only(self, states, mc_returns, epochs=100, batch_size=256, 
                         val_split=0.1, patience=20, target_ev=0.80, lr=1e-5):
        """
        Train only the critic using Monte Carlo returns.
        
        Args:
            states: List of states from collected trajectories
            mc_returns: Corresponding Monte Carlo returns
            epochs: Maximum training epochs
            batch_size: Batch size for training
            val_split: Validation set proportion
            patience: Early stopping patience (epochs without improvement)
            target_ev: Target explained variance
            lr: Learning rate for value training
            
        Returns:
            dict with training metrics (loss history, EV history, etc.)
        """
        if self.verbose:
            print(f"\n=== Phase 2: Value Function Optimization ===")
            print(f"Training samples: {len(states)}")
            print(f"Epochs: {epochs}, Batch size: {batch_size}")
            print(f"Target EV: {target_ev:.2f}, Patience: {patience}")
        
        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(mc_returns, dtype=torch.float32, device=self.device)
        
        # Shuffle and split into train/val
        n_samples = len(states)
        indices = torch.randperm(n_samples)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_states = states_t[train_indices]
        train_returns = returns_t[train_indices]
        val_states = states_t[val_indices]
        val_returns = returns_t[val_indices]
        
        # Create DataLoader for training
        train_dataset = torch.utils.data.TensorDataset(train_states, train_returns)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Create optimizer for critic only
        critic_params = list(self.model.fc1.parameters()) + \
                       list(self.model.fc2.parameters()) + \
                       list(self.model.critic.parameters())
        if hasattr(self.model, 'dropout1'):
            critic_params += list(self.model.dropout1.parameters())
        if hasattr(self.model, 'dropout2'):
            critic_params += list(self.model.dropout2.parameters())
        
        value_optimizer = torch.optim.Adam(critic_params, lr=lr)
        
        # Training loop
        best_val_loss = float('inf')
        best_val_ev = -float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_evs = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            n_batches = 0
            
            for batch_states, batch_returns in train_loader:
                _, value_pred = self.model(batch_states)
                loss = F.mse_loss(value_pred, batch_returns)
                
                value_optimizer.zero_grad()
                loss.backward()
                # Clip gradients for stability
                nn.utils.clip_grad_norm_(critic_params, self.hparams.max_grad_norm)
                value_optimizer.step()
                
                epoch_train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_train_loss / n_batches if n_batches > 0 else 0.0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                _, val_pred = self.model(val_states)
                val_loss = F.mse_loss(val_pred, val_returns)
                
                # Compute explained variance
                var_y = torch.var(val_returns)
                if torch.isfinite(var_y) and var_y > 0:
                    val_ev = 1.0 - torch.var(val_returns - val_pred) / var_y
                    val_ev = float(val_ev.item())
                else:
                    val_ev = 0.0
                
                val_loss_value = float(val_loss.item())
                val_losses.append(val_loss_value)
                val_evs.append(val_ev)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_loss_value:.4f}, "
                      f"Val EV={val_ev:.3f}, "
                      f"Best EV={best_val_ev:.3f}")
            
            # Early stopping logic
            improved = False
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                improved = True
            if val_ev > best_val_ev:
                best_val_ev = val_ev
                improved = True
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check convergence criteria
            if val_ev >= target_ev and patience_counter >= patience:
                if self.verbose:
                    print(f"\n✓ Converged! EV >= {target_ev:.2f} and loss plateaued")
                break
            
            # Early stop on patience alone if loss isn't improving
            if patience_counter >= patience * 2:
                if self.verbose:
                    print(f"\n✓ Stopped: Loss plateaued for {patience*2} epochs")
                break
        
        if self.verbose:
            print(f"\n=== Value Training Complete ===")
            print(f"Final Val Loss: {val_losses[-1]:.4f}")
            print(f"Final Val EV: {val_evs[-1]:.3f}")
            print(f"Best Val EV: {best_val_ev:.3f}")
        
        return {
            'final_val_loss': val_losses[-1] if val_losses else 0.0,
            'final_ev': val_evs[-1] if val_evs else 0.0,
            'best_ev': best_val_ev,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_evs': val_evs,
            'epochs_completed': epoch + 1
        }


