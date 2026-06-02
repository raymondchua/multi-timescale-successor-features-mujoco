from collections import OrderedDict
from typing import TypedDict, List, Dict
from typing_extensions import Unpack

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from absl import logging

class TD3AgentKwargs(TypedDict):
    action_shape: tuple[int, ...]
    batch_size: int
    consolidation: bool
    critic_target_tau: float
    device: str
    domain: str
    feature_dim: int
    hidden_dim: int
    init_critic: bool
    log_grads: bool
    lr: float
    name: str
    nstep: int
    num_expl_steps: int
    obs_shape: tuple[int, ...]
    obs_type: str
    print_grad: bool
    reward_free: bool
    stddev_clip: float
    stddev_schedule: float
    update_encoder: bool
    update_every_steps: int
    use_tb: bool
    use_wandb: bool
    normalize_basis_features_in_critic: bool


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == "pixels" else hidden_dim

        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        policy_layers = []
        policy_layers += [nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)]

        # add additional hidden layer for pixels
        if obs_type == "pixels":
            policy_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]

        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):

        h = self.actor_trunk(obs)
        mu = self.policy(h)

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, normalize_basis_features_in_critic):
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

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]

            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]

            q_layers += [nn.Linear(hidden_dim, 1)]

            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.critic_trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        if self.normalize_basis_features_in_critic:
            h = h / torch.norm(h, dim=1).view((-1, 1))

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class TD3Agent:
    def __init__(
        self,
        **kwargs: Unpack[TD3AgentKwargs]
    ):

        self._kwargs = kwargs
        self.action_dim = self._kwargs["action_shape"][0]
        self.obs_shape = self._kwargs["obs_shape"]
        self.solved_meta = OrderedDict()
        self.device = self._kwargs["device"]
        self.obs_type = self._kwargs["obs_type"]
        self.feature_dim = self._kwargs["feature_dim"]
        self.hidden_dim = self._kwargs["hidden_dim"]
        self.lr = self._kwargs["lr"]
        self.print_grad = self._kwargs["print_grad"]
        self.update_every_steps = self._kwargs["update_every_steps"]
        self.use_tb = self._kwargs["use_tb"]
        self.use_wandb = self._kwargs["use_wandb"]
        self.critic_target_tau = self._kwargs["critic_target_tau"]
        self.batch_size = self._kwargs["batch_size"]
        self.stddev_schedule = self._kwargs["stddev_schedule"]
        self.stddev_clip = self._kwargs["stddev_clip"]
        self.meta_dim = self._kwargs["meta_dim"]
        self.update_encoder = self._kwargs["update_encoder"]
        self.normalize_basis_features_in_critic = self._kwargs["normalize_basis_features_in_critic"]
        self._consolidation = self._kwargs["consolidation"]


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
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim, self.normalize_basis_features_in_critic
        ).to(self.device)

        self.critic_target = Critic(
            self.obs_type, self.obs_dim, self.action_dim, self.feature_dim, self.hidden_dim, self.normalize_basis_features_in_critic
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

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self._kwargs["init_critic"]:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode, num_valid_actions):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        value_normalized = None
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            value = F.normalize(value, p=2, dim=-1)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)

        # assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self._kwargs["stddev_schedule"], step)

        dist = self.actor(inpt, stddev)

        if eval_mode:
            action = dist.mean

        else:
            action = dist.sample(clip=None)
            if step < self._kwargs["num_expl_steps"]:
                action.uniform_(-1.0, 1.0)

        # make sure the number of actions is valid as per the environment
        action = action[:num_valid_actions]

        return action.cpu().numpy()[0]

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
        Q1, Q2 = self.critic(obs, action)
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
        return self.encoder(obs)

    def update(self, replay_iter, step):
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
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self._kwargs["critic_target_tau"]
        )

        self._update_step += 1

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["batch_reward"] = reward.mean().item()
            metrics["update_step"] = self._update_step

        return metrics

    @torch.no_grad()
    def solved_meta(self):
        return None

    @torch.no_grad()
    def num_params(self):
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        return sum(p.numel() for p in all_params)

    @torch.no_grad()
    def rotate_nd_vector_fixed_angle(self, action_vec, angle_degrees):
        """
        Rotate an n-dimensional action_vec by a fixed angle across all planes.

        Args:
        - action_vec (torch.Tensor): Input action_vec of shape (n,).
        - angle_degrees (float): Fixed angle for all planes of rotation.

        Returns:
        - torch.Tensor: Rotated n-dimensional action_vec.
        """
        n = action_vec.shape[0]
        combined_rotation_matrix = torch.eye(n).to(action_vec.device)

        # Compute rotation matrix for each pair of axes
        for axis1 in range(n):
            for axis2 in range(axis1 + 1, n):
                # Generate the rotation matrix for the current plane
                rotation_matrix = torch.eye(n).to(action_vec.device)
                angle = torch.radians(torch.tensor(angle_degrees, dtype=torch.float32).to(action_vec.device))

                rotation_matrix[axis1, axis1] = torch.cos(angle)
                rotation_matrix[axis2, axis2] = torch.cos(angle)
                rotation_matrix[axis1, axis2] = -torch.sin(angle)
                rotation_matrix[axis2, axis1] = torch.sin(angle)

                # Combine with the previous rotations
                combined_rotation_matrix = torch.matmul(rotation_matrix, combined_rotation_matrix)

        # Apply the combined rotation matrix to the action vector
        rotated_vector = torch.matmul(combined_rotation_matrix, action_vec)
        return

    @property
    def consolidation(self):
        return self._consolidation

    @property
    def name(self):
        return self._kwargs["name"]

    def update_every_steps(self):
        return self.update_every_steps

    @property
    def update_step(self):
        return self._update_step

    def get_named_params_list(
        self, models: List[torch.nn.Module]
    ) -> List[Dict[str, torch.Tensor]]:
        return [dict(model.named_parameters()) for model in models]

    @property
    def use_plasticity_injection(self):
        return False