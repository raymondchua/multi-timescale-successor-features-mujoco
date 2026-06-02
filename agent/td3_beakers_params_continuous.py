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

class TD3BeakersParamsContinuousKwargs(TD3AgentKwargs):
    beaker_capacity: float
    flow_factor: int
    lr_consolidation: float
    max_grad_norm_consolidation: float
    num_beakers: int
    update_consolidation_every_steps: int

class TD3_beakers_params_continuous(TD3Agent):
    def __init__(
        self,
        **kwargs: Unpack[TD3BeakersParamsContinuousKwargs],
    ):
        super().__init__(
            **kwargs,
        )

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

    def set_up_consolidation_system(self):

        self.g_flow = 0.1/ self.capacity[1]
        self.storage_timescales = torch.zeros(
            self.num_beakers, device=self.device, dtype=torch.int
        )
        self.recall_timescales = torch.zeros(
            self.num_beakers, device=self.device, dtype=torch.int
        )

        self.scale_consolidation = torch.zeros(
            self.num_beakers, device=self.device, dtype=torch.float
        )

        self.scale_recall = torch.zeros(
            self.num_beakers, device=self.device, dtype=torch.float
        )

        for exp in range(self.num_beakers):

            self.storage_timescales[exp] = math.ceil(
                (
                    self.capacity[exp]
                    / (self.g_flow * self.lr_consolidation)
                ) * self.update_consolidation_every_steps
            )

            self.recall_timescales[exp] = self.storage_timescales[exp]

            self.scale_consolidation[exp] =  self.g_flow / self.capacity[exp]
            self.scale_recall[exp] = self.g_flow / self.capacity[exp + 1]


        logging.info(f"g_flow: {self.g_flow}")
        logging.info(f"Capacity: {self.capacity}")
        logging.info(f"storage g_flow: {self.g_flow}")
        logging.info(f"recall g_flow: {self.g_flow}")
        logging.info(f"storage timescales: {self.storage_timescales}")
        logging.info(f"recall timescales: {self.recall_timescales}")
        logging.info(f"scale consolidation: {self.scale_consolidation}")
        logging.info(f"scale recall: {self.scale_recall}")

        assert (
            self.g_flow * self.update_consolidation_every_steps <= 0.1
        ), "g_flow * update_consolidation_every_steps should be less than or equal to 0.1"

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        mask_condition_recall = self.compute_recall_mask(
            self.update_step, self.recall_timescales
        )

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # for current obs, we only aug as we will encode in the update_critic function since there is a different encoder
        # for each beaker
        obs = self.aug_and_encode(obs)
        # obs_all = [obs]

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        if self._update_step % self.update_consolidation_every_steps == 0:
            if self.obs_type == "pixels":
                # update the parameters of the encoder networks
                (
                    critic_consolidation_loss,
                    encoder_consolidation_loss,
                    critic_params_norm,
                    encoder_params_norm,
                ) = self.consolidation_update_fn(
                    self.g_flow,
                    self.capacity,
                    self.num_beakers,
                    mask_condition_recall,
                )

                for i in range(self.num_beakers):
                    metrics[f"encoder_params_norm_{i}"] = encoder_params_norm[i].item()

            else:
                (
                    critic_consolidation_loss,
                    critic_params_norm,
                ) = self.consolidation_update_fn(
                    self.g_flow,
                    self.capacity,
                    self.num_beakers,
                    mask_condition_recall,
                )

                encoder_consolidation_loss = 0.0

            if self.use_tb or self.use_wandb:
                for i in range(self.num_beakers):
                    metrics[f"critic_params_norm_{i}"] = critic_params_norm[
                        i
                    ].item()

                    metrics[f"mask_condition_recall_{i}"] = mask_condition_recall[
                        i
                    ].item()

                    metrics["critic_consolidation_loss"] = critic_consolidation_loss[
                        i
                    ].item()

                    if self.obs_type == "pixels":
                        metrics[
                            "encoder_consolidation_loss"
                        ] = encoder_consolidation_loss[i].item()
                    else:
                        metrics["encoder_consolidation_loss"] = 0.0

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic_networks[0], self.critic_target, self.critic_target_tau
        )

        self._update_step += 1

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["batch_reward"] = reward.mean().item()
            metrics["update_step"] = self._update_step

        if self._kwargs["log_grad"]:
            # print keys and values in metrics for debugging

            if self._kwargs["print_grad"]:
                if metrics is not None:
                    for k, v in metrics.items():
                        print(f"{k}: {v}")

            for i in range(self.num_beakers):
                prefix = "network_" + str(i) + "_"
                if self._kwargs["print_grad"]:
                    print("network: ", i)

                for name, param in self.critic_networks[i].named_parameters():
                    if param.grad is not None:
                        name_grad = prefix + name + "_grad"
                        metrics[name_grad] = param.grad.norm()
                        if self._kwargs["print_grad"]:
                            print(name_grad, param.grad.norm())
                        # metrics[name] = param.norm()
                        # print(name, param.norm())
                    else:
                        name_grad = prefix + name + "_grad"
                        if self._kwargs["print_grad"]:
                            print(name_grad, "None")

                    name_param_norm = prefix + name + "_norm"
                    metrics[name_param_norm] = param.norm()
                    if self._kwargs["print_grad"]:
                        print(name_param_norm, param.norm())

                if self.obs_type == "pixels":
                    for name, param in self.encoder_networks[i].named_parameters():
                        if param.grad is not None:
                            name_grad = prefix + name + "_grad"
                            metrics[name_grad] = param.grad.norm()
                            if self._kwargs["print_grad"]:
                                print(name_grad, param.grad.norm())
                            # metrics[name] = param.norm()
                            # print(name, param.norm())
                        else:
                            name_grad = prefix + name + "_grad"
                            if self._kwargs["print_grad"]:
                                print(name_grad, "None")

            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    name_grad = name + "_grad"
                    metrics[name_grad] = param.grad.norm()
                    if self._kwargs["print_grad"]:
                        print(name_grad, param.grad.norm())
                    # metrics[name] = param.norm()
                    # print(name, param.norm())
                else:
                    name_grad = name + "_grad"
                    if self._kwargs["print_grad"]:
                        print(name_grad, "None")

                name_param_norm = name + "_norm"
                metrics[name_param_norm] = param.norm()
                if self._kwargs["print_grad"]:
                    print(name_param_norm, param.norm())

        return metrics

    def consolidation_update_fn(self, g_flow, capacity, num_beakers, mask):
        critic_loss = enc_loss = torch.zeros(num_beakers, device=self.device)

        # Initialize
        critic_norms = torch.zeros(num_beakers, device=self.device)
        enc_norms = torch.zeros(num_beakers, device=self.device)

        # Extract named params
        critic_params = self.get_named_params_list(self.critic_networks)
        enc_params = self.get_named_params_list(self.encoder_networks)

        # Recall from 2nd to 1st beaker
        scale_first = g_flow / capacity[1]
        (
            critic_loss,
            enc_loss,
            critic_params,
            enc_params,
        ) = self._consolidate_all_networks(
            0,
            1,
            scale_first * self.lr_consolidation,
            mask[1] * self.update_consolidation_every_steps,
            critic_params,
            enc_params,
            critic_loss,
            enc_loss,
        )

        critic_norms[0] = self.compute_param_norms(critic_params[0])
        if self.obs_type == "pixels":
            enc_norms[0] = self.compute_param_norms(enc_params[0])

        # Loop over remaining beakers
        for i in range(1, num_beakers):
            # Forward consolidation from prev
            scale_prev = g_flow / capacity[i]
            (
                critic_loss,
                enc_loss,
                critic_params,
                enc_params,
            ) = self._consolidate_all_networks(
                i,
                i - 1,
                scale_prev * self.lr_consolidation,
                self.update_consolidation_every_steps,
                critic_params,
                enc_params,
                critic_loss,
                enc_loss,
            )

            # Optional recall from next
            if i < num_beakers - 1 and mask[i + 1] != 0:
                scale_next = g_flow / capacity[i + 1]
                (
                    critic_loss,
                    enc_loss,
                    critic_params,
                    enc_params,
                ) = self._consolidate_all_networks(
                    i,
                    i + 1,
                    scale_next * self.lr_consolidation,
                    mask[i + 1] * self.update_consolidation_every_steps,
                    critic_params,
                    enc_params,
                    critic_loss,
                    enc_loss,
                )
            elif i == num_beakers - 1:
                scale_last = g_flow / capacity[i + 1]
                (
                    critic_loss,
                    enc_loss,
                    critic_params,
                    enc_params,
                ) = self._consolidate_all_networks(
                    i,
                    None,
                    scale_last * self.lr_consolidation,
                    self.update_consolidation_every_steps,
                    critic_params,
                    enc_params,
                    critic_loss,
                    enc_loss,
                )

            # Norms
            critic_norms[i] = self.compute_param_norms(critic_params[i])
            if self.obs_type == "pixels":
                enc_norms[i] = self.compute_param_norms(enc_params[i])

        # Copy back params
        for i in range(num_beakers):
            for name, p in self.critic_networks[i].named_parameters():
                p.data.copy_(critic_params[i][name])
            if self.obs_type == "pixels":
                for name, p in self.encoder_networks[i].named_parameters():
                    p.data.copy_(enc_params[i][name])

        if self.obs_type == "pixels":
            return critic_loss, enc_loss, critic_norms, enc_norms
        else:
            return critic_loss, critic_norms

    def compute_recall_mask(
        self, update_step: int, recall_timescales: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the recall mask based on a fixed recall_timescales and the current update step.

        Args:
            update_step: scalar int
            recall_timescales: 1D tensor (e.g. torch.tensor([0, 100, 200, ...]))

        Returns:
            mask: binary (int32) mask of shape (num_beakers,)
        """
        mask = (recall_timescales < update_step).to(dtype=torch.int32)
        mask = torch.cat(
            [torch.tensor([1], dtype=torch.int32, device=mask.device), mask[:-1]]
        )
        return mask

    def compute_param_norms(self, param_dict):
        return sum(p.norm(p=2).pow(2).sqrt() for p in param_dict.values())

    def _consolidate_all_networks(
        self,
        i,
        j,
        scale,
        masks,
        critic_params,
        enc_params,
        critic_loss,
        enc_loss,
    ):
        if j is not None:
            critic_src = critic_params[j]
            enc_src = enc_params[j] if self.obs_type == "pixels" else None
        else:
            critic_src = self.critic_params_set_to_zero
            enc_src = (
                self.encoder_params_set_to_zero if self.obs_type == "pixels" else None
            )

        critic_params[i], critic_loss[i] = self.update_and_accumulate_tree_flat(
            critic_params[i], critic_src, scale, masks
        )

        if self.obs_type == "pixels":
            enc_params[i], enc_loss[i] = self.update_and_accumulate_tree_flat(
                enc_params[i], enc_src, scale, masks
            )

        return critic_loss, enc_loss, critic_params, enc_params

    def update_and_accumulate_tree_flat(
        self,
        p1: Dict[str, torch.Tensor],
        p2: Dict[str, torch.Tensor],
        scale: float,
        update_consolidation_every_steps: float = 1.0,
        mask: int = 1,
        max_norm: float = 10.0,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        scale_factor = scale * mask * update_consolidation_every_steps

        # Flatten
        vec_a = parameters_to_vector(p1.values())
        vec_b = parameters_to_vector(p2.values())

        delta = vec_b - vec_a
        delta_scaled = delta * scale_factor

        norm_sq = delta_scaled.pow(2).sum()
        norm = norm_sq.sqrt()
        clip_factor = torch.clamp(max_norm / (norm + 1e-6), max=1.0)

        vec_updated = vec_a + delta_scaled * clip_factor

        # Map back to parameter dict shape
        updated_values = list(p1.values())  # shape
        vector_to_parameters(vec_updated, updated_values)
        updated_dict = {k: v for k, v in zip(p1.keys(), updated_values)}

        return updated_dict, norm_sq

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self._kwargs["stddev_clip"])
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic_networks[0](obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self._kwargs["stddev_schedule"], step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self._kwargs["stddev_clip"])
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic_networks[0](obs, action)
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

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder_networks[0](obs)

    @torch.no_grad()
    def num_params(self):
        all_params = list(self.actor.parameters())

        for i in range(self.num_beakers):
            all_params += list(self.critic_networks[i].parameters())

            if self.obs_type == "pixels":
                all_params += list(self.encoder_networks[i].parameters())

        num_parameters = sum([params.numel() for params in all_params])

        return num_parameters
