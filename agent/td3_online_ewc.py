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

class TD3OnlineEWCKwargs(TD3AgentKwargs):
    ewc_gamma: float
    ewc_regularization: float
    fisher_update_interval: int

class TD3_online_EWC(TD3Agent):
    def __init__(
        self,
        **kwargs: Unpack[TD3OnlineEWCKwargs],
    ):
        super().__init__(
            **kwargs,
        )

        self.ewc_gamma = self._kwargs["ewc_gamma"]
        self.ewc_regularization = self._kwargs["ewc_regularization"]
        self.fisher_update_interval = self._kwargs["fisher_update_interval"]

        self.params_star = None  # Previous parameter snapshot
        self.fisher = None  # Running Fisher estimate

    def ewc_penalty(self):
        if self.params_star is None or self.fisher is None:
            return torch.tensor(0.0, device=self.device)

        penalty = 0.0
        for p, p_star, f in zip(self.critic.parameters(), self.params_star, self.fisher):
            penalty += torch.sum(f * (p_star - p) **2)

        return 0.5 * self.ewc_regularization * penalty

    def compute_fisher(self, loss):
        """Compute squared gradients and accumulate into running Fisher."""
        grads = torch.autograd.grad(loss, self.critic.parameters(), retain_graph=True)
        fisher_batch = [g.detach() ** 2 for g in grads]

        if self.fisher is None:
            self.fisher = fisher_batch
        else:
            self.fisher = [
                self.ewc_gamma * f_old + f_new
                for f_old, f_new in zip(self.fisher, fisher_batch)
            ]

    def consolidate(self):
        self.params_star = [p.detach().clone() for p in self.critic.parameters()]


    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self._kwargs["stddev_clip"])
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        ewc_loss = self.ewc_penalty()
        critic_loss_with_penalty = critic_loss + ewc_loss

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["ewc_loss"] = ewc_loss.item()
            metrics["critic_loss_with_penalty"] = critic_loss_with_penalty.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_with_penalty.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if step % self._kwargs["fisher_update_interval"] == 0:
            with torch.no_grad():
                stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self._kwargs["stddev_clip"])
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                target_V = torch.min(target_Q1, target_Q2)
                target_Q = reward + (discount * target_V)

            Q1, Q2 = self.critic(obs, action)
            critic_loss_no_ewc = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            self.compute_fisher(critic_loss_no_ewc)
            self.consolidate()

        return metrics

