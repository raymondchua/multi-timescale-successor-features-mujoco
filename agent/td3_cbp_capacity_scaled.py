from collections import OrderedDict
from typing import TypedDict, List, Dict, Optional
from typing_extensions import Unpack

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils

from absl import logging

from agent.td3_cbp import (
    TD3_continual_backprop_Agent,
    TD3_continual_backprop_AgentKwargs,
    ActivationsTracker,
    Actor,
    Critic,
)


class TD3_continual_backprop_capacity_scaled_AgentKwargs(
    TD3_continual_backprop_AgentKwargs
):
    capacity_scale: int


class TD3_continual_backprop_capcity_scaled_Agent(TD3_continual_backprop_Agent):
    def __init__(
        self, **kwargs: Unpack[TD3_continual_backprop_capacity_scaled_AgentKwargs]
    ):

        super().__init__(**kwargs)
        self.capacity_scale = self._kwargs["capacity_scale"]
        self.feature_dim = int(self.capacity_scale * self.feature_dim)
        self.hidden_dim = int(self.capacity_scale * self.hidden_dim)

        # models

        self.actor = Actor(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self._kwargs["ema_decay"],
        ).to(self.device)

        self.critic = Critic(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self.normalize_basis_features_in_critic,
            self._kwargs["ema_decay"],
        ).to(self.device)

        self.critic_target = Critic(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self.normalize_basis_features_in_critic,
            self._kwargs["ema_decay"],
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.obs_type == "pixels":
            self.ACTOR_OUT_MAP = {
                "actor_trunk.0": "policy.0",  # trunk CBP → first policy CBP
                "policy.0": "policy.2",  # skip ReLU; next CBP
                "policy.2": "policy.4",  # skip ReLU; next CBP (final head is policy.4; we won't refresh it)
            }

            self.CRITIC_OUT_MAP = {
                # trunk feeds both Q1.0 and Q2.0 (fan-out)
                "critic_trunk.0": ["Q1.0", "Q2.0"],
                # within each Q stream
                "Q1.0": "Q1.2",
                "Q1.2": "Q1.4",  # final head is Q1.4; we won't refresh it
                "Q2.0": "Q2.2",
                "Q2.2": "Q2.4",  # final head is Q2.4; we won't refresh it
            }

            self.actor_layers_for_rank = [
                "actor_trunk.0",  # LinearCBP(obs_dim, feature_dim)
                "policy.0",  # LinearCBP(feature_dim, hidden_dim)
                "policy.2",  # LinearCBP(hidden_dim, hidden_dim)
                # (skip the final head: policy.4 = LinearCBP(hidden_dim, action_dim))
            ]

            self.critic_layers_for_rank = [
                "critic_trunk.0",  # LinearCBP(obs_dim, feature_dim)
                "Q1.0",  # LinearCBP(feature_dim+action_dim, hidden_dim)
                "Q1.2",  # LinearCBP(hidden_dim, hidden_dim)
                "Q2.0",  # LinearCBP(feature_dim+action_dim, hidden_dim)
                "Q2.2",  # LinearCBP(hidden_dim, hidden_dim)
                # (skip the final heads: Q1.4/Q2.4 = LinearCBP(hidden_dim, 1))
            ]

        else:
            self.ACTOR_OUT_MAP = {
                "actor_trunk.0": "policy.0",
                "policy.0": "policy.2",  # final head is policy.2; we won't refresh it
            }

            self.CRITIC_OUT_MAP = {
                "critic_trunk.0": ["Q1.0", "Q2.0"],
                "Q1.0": "Q1.2",  # final head is Q1.2; we won't refresh it
                "Q2.0": "Q2.2",
            }

            self.actor_layers_for_rank = [
                "actor_trunk.0",  # LinearCBP(obs_dim, hidden_dim)
                "policy.0",  # LinearCBP(hidden_dim, hidden_dim)
                # (skip the final head: policy.2 = LinearCBP(hidden_dim, action_dim))
            ]

            self.critic_layers_for_rank = [
                "critic_trunk.0",  # LinearCBP(obs_dim + action_dim, hidden_dim)
                "Q1.0",  # LinearCBP(hidden_dim, hidden_dim)
                "Q2.0",  # LinearCBP(hidden_dim, hidden_dim)
                # (skip the final heads: Q1.2/Q2.2 = LinearCBP(hidden_dim, 1))
            ]

        # optimizers

        if self.obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(
                self.encoder.parameters(), lr=self._kwargs["lr"]
            )
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self._kwargs["lr"]
        )

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self._kwargs["lr"]
        )

        self._update_step = 0

        self.actor_tracker = ActivationsTracker(self.actor, self.actor_layers_for_rank)
        self.critic_tracker = ActivationsTracker(
            self.critic, self.critic_layers_for_rank
        )

        self.train()
        self.critic_target.train()

        self.actor_cbp_layers = [
            n
            for n, m in self.actor.named_modules()
            if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")
        ]
        self.critic_cbp_layers = [
            n
            for n, m in self.critic.named_modules()
            if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")
        ]
