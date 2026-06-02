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

from agent.td3_beakers_params_continuous import TD3_beakers_params_continuous, TD3BeakersParamsContinuousKwargs, Encoder, Critic
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class TD3BeakersParamsContinuousCapacityScaledKwargs(TD3BeakersParamsContinuousKwargs):
    capacity_scale: int

class TD3_beakers_params_continuous_capacity_scaled(TD3_beakers_params_continuous):
    def __init__(
        self,
        **kwargs: Unpack[TD3BeakersParamsContinuousKwargs],
    ):
        super().__init__(
            **kwargs,
        )
        self.capacity_scale = self._kwargs["capacity_scale"]
        self.feature_dim = int(self.capacity_scale * self.feature_dim)
        self.hidden_dim = int(self.capacity_scale * self.hidden_dim)

        self.num_beakers = self._kwargs["num_beakers"]
        self.beaker_capacity = self._kwargs["beaker_capacity"]
        self.max_grad_norm_consolidation = self._kwargs["max_grad_norm_consolidation"]
        self.log_grad = self._kwargs["log_grad"]
        self.flow_factor = self._kwargs["flow_factor"]
        self.lr_consolidation = self._kwargs["lr_consolidation"]
        self.update_consolidation_every_steps = self._kwargs[
            "update_consolidation_every_steps"
        ]
        self._num_train_frames = None

        self.encoder_networks = []
        self.critic_networks = []
        self.capacity = torch.zeros(
            self.num_beakers + 1, device=self.device, dtype=torch.int
        ) # additional beaker for the auxiliary zero beaker at the end

        for exp in range(self.num_beakers):

            if exp == 0:
                self.capacity[exp] = 1

            else:
                self.capacity[exp] = (self.beaker_capacity ** (exp)) * self.flow_factor

            critic = Critic(
                self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim, self.normalize_basis_features_in_critic
            ).to(self.device)

            if self.obs_type == "pixels":
                encoder =  Encoder(self.obs_shape).to(self.device)

            else:
                encoder = nn.Identity()

            self.critic_networks.append(critic)
            self.encoder_networks.append(encoder)

        # add an additional beaker for the auxiliary zero beaker as last beaker
        self.capacity[self.num_beakers] = (self.beaker_capacity ** (self.num_beakers)) * self.flow_factor

        # only one critic target since the target is only used for the first beaker
        self.critic_target = Critic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim,
            self.normalize_basis_features_in_critic
        ).to(self.device)

        critic_params: List[Dict[str, torch.Tensor]] = self.get_named_params_list(
            self.critic_networks
        )

        for i in range(self.num_beakers):
            network_i_params_norm = torch.zeros(self.num_beakers).to(self.device)
            for name, param in critic_params[i].items():
                network_i_params_norm[i] += param.norm()

        self.critic_opt = torch.optim.Adam(
            self.critic_networks[0].parameters(), lr=self.lr
        )
        self.critic_target.load_state_dict(self.critic_networks[0].state_dict())

        if self.obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(
                self.encoder_networks[0].parameters(), lr=self.lr
            )

        self.critic_params_set_to_zero = utils.get_params_set_to_zero(
            self.critic_networks[0]
        )

        if self.obs_type == "pixels":
            self.encoder_params_set_to_zero = utils.get_params_set_to_zero(
                self.encoder_networks[0]
            )

        self.critic_target.train()
        self.actor.train(True)
        for i in range(self.num_beakers):
            self.critic_networks[i].train(True)

            if self.obs_type == "pixels":
                self.encoder_networks[i].train(True)