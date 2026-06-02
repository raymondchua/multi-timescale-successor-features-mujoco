from collections import OrderedDict
from typing import Dict, Tuple, List, Optional
from typing_extensions import Unpack

import math
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from absl import logging

from agent.td3 import TD3Agent, TD3AgentKwargs, Encoder, Critic, Actor

class TD3CapacityScaleKwargs(TD3AgentKwargs):
    capacity_scale: int

class TD3_capacity_scaled(TD3Agent):
    def __init__(
        self,
        **kwargs: Unpack[TD3CapacityScaleKwargs],
    ):
        super().__init__(
            **kwargs,
        )

        self.capacity_scale = self._kwargs["capacity_scale"]
        self.feature_dim = int(self.capacity_scale * self.feature_dim)
        self.hidden_dim = int(self.capacity_scale * self.hidden_dim)

        # models
        if self.obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(self.obs_shape).to(self.device)
            self.obs_dim = self.encoder.repr_dim + self.meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim

        self.actor = Actor(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim
        ).to(self.device)

        self.critic = Critic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim,
            self.normalize_basis_features_in_critic
        ).to(self.device)

        self.critic_target = Critic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim,
            self.normalize_basis_features_in_critic
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if self.obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self._kwargs["lr"])
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self._kwargs["lr"])

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self._kwargs["lr"])

        self._update_step = 0

        self.train()
        self.critic_target.train()

