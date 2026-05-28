"""Agents package for Flappy Bird Reinforcement Learning"""

from . import ppo_agent
from .ppo_agent import PPO_agent

__all__ = ["PPO_agent", "ppo_agent"]