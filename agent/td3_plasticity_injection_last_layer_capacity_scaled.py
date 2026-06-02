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

from agent.td3_plasticity_injection_last_layer import TD3_plasticity_injection_last_layer_Agent, Critic_encoder, Critic_head, TD3AgentKwargs, Encoder

class TD3BeakersParamsContinuousCapacityScaledKwargs(TD3AgentKwargs):
    capacity_scale: int

class TD3_plasticity_injection_last_layer_capacity_scaled_Agent(TD3_plasticity_injection_last_layer_Agent):
    def __init__(self, **kwargs: Unpack[TD3BeakersParamsContinuousCapacityScaledKwargs]):
        super().__init__(
            **kwargs,
        )

        self.capacity_scale = self._kwargs["capacity_scale"]
        self.feature_dim = int(self.capacity_scale * self.feature_dim)
        self.hidden_dim = int(self.capacity_scale * self.hidden_dim)

        self.action_dim = self._kwargs["action_shape"][0]
        self.obs_shape = self._kwargs["obs_shape"]
        self.device = self._kwargs["device"]
        self.obs_type = self._kwargs["obs_type"]
        self.meta_dim = self._kwargs["meta_dim"]

        # models
        if self.obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(self.obs_shape).to(self.device)
            self.obs_dim = self.encoder.repr_dim + self.meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim

        self.critic_encoder = Critic_encoder(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self._kwargs["normalize_basis_features_in_critic"],
        ).to(self.device)

        self.critic_head = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        self.critic_head_random1 = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        self.critic_head_random2 = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        """
        Target networks
        """

        self.critic_encoder_target = Critic_encoder(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self._kwargs["normalize_basis_features_in_critic"],
        ).to(self.device)

        self.critic_head_target = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        self.critic_head_random1_target = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        self.critic_head_random2_target = Critic_head(
            self.obs_type,
            self.hidden_dim,
        ).to(self.device)

        """
        Loading params for target networks
        """

        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        self.critic_head_target.load_state_dict(self.critic_head.state_dict())
        self.critic_head_random1_target.load_state_dict(
            self.critic_head_random1.state_dict()
        )
        self.critic_head_random2_target.load_state_dict(
            self.critic_head_random2.state_dict()
        )

        self.critic_head_random1.load_state_dict(self.critic_head.state_dict())
        self.critic_head_random2.load_state_dict(self.critic_head.state_dict())

        """
        Optimizers
        """

        self.critic_encoder_opt = torch.optim.Adam(
            self.critic_encoder.parameters(), lr=self._kwargs["lr"]
        )
        self.critic_head_opt = torch.optim.Adam(
            self.critic_head.parameters(), lr=self._kwargs["lr"]
        )
        self.critic_head_random1_opt = torch.optim.Adam(
            self.critic_head_random1.parameters(), lr=self._kwargs["lr"]
        )
        self.critic_head_random2_opt = torch.optim.Adam(
            self.critic_head_random2.parameters(), lr=self._kwargs["lr"]
        )

        self.train()