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

from agent.td3 import TD3Agent, TD3AgentKwargs, Encoder, Critic
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Critic_encoder(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_dim,
        action_dim,
        feature_dim,
        hidden_dim,
        normalize_basis_features_in_critic,
    ):
        super().__init__()

        self.obs_type = obs_type
        self.normalize_basis_features_in_critic = normalize_basis_features_in_critic

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.critic_trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )

            trunk_dim = feature_dim + action_dim

        else:
            # for states actions come in the beginning
            self.critic_trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )

            trunk_dim = hidden_dim

        def make_last_few_layers():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]

            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]

            return nn.Sequential(*q_layers)

        self.last_few_layers_1 = make_last_few_layers()
        self.last_few_layers_2 = make_last_few_layers()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.critic_trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        if self.normalize_basis_features_in_critic:
            h = h / torch.norm(h, dim=1).view((-1, 1))

        h1 = self.last_few_layers_1(h)
        h2 = self.last_few_layers_2(h)

        return h1, h2


class Critic_head(nn.Module):
    def __init__(
        self,
        obs_type,
        hidden_dim,
    ):
        super().__init__()

        self.obs_type = obs_type


        self.last_layer = nn.Linear(hidden_dim, 1)

        self.apply(utils.weight_init)

    def forward(self, h1, h2):

        q1 = self.last_layer(h1)
        q2 = self.last_layer(h2)

        return q1, q2


class TD3_plasticity_injection_last_layer_Agent(TD3Agent):
    def __init__(self, **kwargs: Unpack[TD3AgentKwargs]):

        self._kwargs = kwargs
        self.action_dim = self._kwargs["action_shape"][0]
        self.obs_shape = self._kwargs["obs_shape"]
        self.device = self._kwargs["device"]
        self.obs_type = self._kwargs["obs_type"]
        self.feature_dim = self._kwargs["feature_dim"]
        self.hidden_dim = self._kwargs["hidden_dim"]
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

        super().__init__(
            **kwargs,
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic_encoder.train(training)
        self.critic_head.train(training)
        self.critic_head_random1.train(training)
        self.critic_head_random2.train(training)

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, plasticity_injection=False
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self._kwargs["stddev_clip"])
            next_h1, next_h2 = self.critic_encoder_target(next_obs, next_action)
            target_Q1, target_Q2 = self.critic_head_target(next_h1, next_h2)

            if plasticity_injection:
                target_Q1_random1, target_Q2_random1 = self.critic_head_random1_target(
                    next_h1, next_h2
                )
                target_Q1_random2, target_Q2_random2 = self.critic_head_random2_target(
                    next_h1, next_h2
                )
                combined_target_Q1 = target_Q1 + target_Q1_random1 - target_Q1_random2
                combined_target_Q2 = target_Q2 + target_Q2_random1 - target_Q2_random2
                target_V = torch.min(combined_target_Q1, combined_target_Q2)

            else:
                target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        h1, h2 = self.critic_encoder(obs, action)
        Q1, Q2 = self.critic_head(h1, h2)
        if plasticity_injection:
            Q1_random_1, Q2_random_1 = self.critic_head_random1(h1, h2)
            Q1_random_2, Q2_random_2 = self.critic_head_random2(h1, h2)
            combined_Q1 = Q1.detach() + Q1_random_1 - Q1_random_2.detach()
            combined_Q2 = Q2.detach() + Q2_random_1 - Q2_random_2.detach()
            critic_loss = F.mse_loss(combined_Q1, target_Q) + F.mse_loss(
                combined_Q2, target_Q
            )

        else:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_encoder_opt.zero_grad(set_to_none=True)
        self.critic_head_opt.zero_grad(set_to_none=True)
        self.critic_head_random1_opt.zero_grad(set_to_none=True)
        self.critic_head_random2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_encoder_opt.step()
        self.critic_head_opt.step()
        self.critic_head_random1_opt.step()
        self.critic_head_random2_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, plasticity_injection=False):
        metrics = dict()

        stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self._kwargs["stddev_clip"])
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        h1, h2 = self.critic_encoder(obs, action)
        Q1, Q2 = self.critic_head(h1, h2)
        if plasticity_injection:
            Q1_random_1, Q2_random_1 = self.critic_head_random1(h1, h2)
            Q1_random_2, Q2_random_2 = self.critic_head_random2(h1, h2)
            Q1 = Q1.detach() + Q1_random_1 - Q1_random_2.detach()
            Q2 = Q2.detach() + Q2_random_1 - Q2_random_2.detach()

        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step, plasticity_injection=False):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self._kwargs["update_every_steps"] != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        # update critic
        metrics.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                step,
                plasticity_injection=plasticity_injection,
            )
        )

        # update actor
        metrics.update(
            self.update_actor(obs.detach(), step, plasticity_injection=plasticity_injection)
        )

        # update critic target
        utils.soft_update_params(
            self.critic_encoder,
            self.critic_encoder_target,
            self._kwargs["critic_target_tau"],
        )

        utils.soft_update_params(
            self.critic_head, self.critic_head_target, self._kwargs["critic_target_tau"]
        )

        utils.soft_update_params(
            self.critic_head_random1,
            self.critic_head_random1_target,
            self._kwargs["critic_target_tau"],
        )

        utils.soft_update_params(
            self.critic_head_random2,
            self.critic_head_random2_target,
            self._kwargs["critic_target_tau"],
        )

        self._update_step += 1

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["batch_reward"] = reward.mean().item()
            metrics["update_step"] = self._update_step

        return metrics

    @torch.no_grad()
    def num_params(self):
        all_params = (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_head.parameters())
            + list(self.critic_head_random1.parameters())
            + list(self.critic_head_random2.parameters())
        )

        return sum(p.numel() for p in all_params)

    def use_plasticity_injection(self):
        return True