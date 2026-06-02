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

from agent.td3 import TD3Agent, TD3AgentKwargs


class TD3_continual_backprop_AgentKwargs(TD3AgentKwargs):
    dead_eps: float
    ema_decay: float
    maturity_threshold: int
    bottom_frac: float


"""
This is the continual backprop variant of td3
"""


class ActivationsTracker:
    """Registers forward hooks on chosen modules and stores last-batch activations."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = set(layer_names)
        self.buffers: Dict[str, torch.Tensor] = {}
        self._hooks = []
        named = dict(model.named_modules())
        for name in layer_names:
            if name not in named:
                raise KeyError(f"{name!r} not found in model.named_modules().")
            mod = named[name]
            h = mod.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _make_hook(self, name):
        def _hook(mod, inp, out):
            # Store as (B, U) matrix; if needed, flatten trailing dims
            x = out
            if x.dim() > 2:
                x = x.flatten(start_dim=1)
            self.buffers[name] = x.detach()

        return _hook

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class LinearCBP(nn.Linear):
    """nn.Linear with per-unit EMA utility + age buffers."""

    def __init__(self, in_features, out_features, bias=True, ema_decay=0.99):
        super().__init__(in_features, out_features, bias=bias)
        self.ema_decay = ema_decay
        self.register_buffer("utility_ema", torch.zeros(out_features))
        self.register_buffer("age", torch.zeros(out_features, dtype=torch.long))
        self.register_buffer("cbp_counter", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "cbp_resets", torch.zeros((), dtype=torch.long)
        )  # count resets for logging

    @torch.no_grad()
    def tick_age(self):
        self.age.add_(1)

    def forward(self, x, record_utility: bool = True):
        y = F.linear(x, self.weight, self.bias)  # [B, out]
        if record_utility and self.training:
            with torch.no_grad():
                # proxy for contribution utility (cheap, batch-level):
                # |activation| averaged across batch
                u = y.abs().mean(dim=0)  # [out]
                self.utility_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * u)
        return y


class Actor(nn.Module):
    def __init__(
        self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, ema_decay
    ):
        super().__init__()

        feature_dim = feature_dim if obs_type == "pixels" else hidden_dim

        self.actor_trunk = nn.Sequential(
            LinearCBP(obs_dim, feature_dim, ema_decay=ema_decay),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        policy_layers = []
        policy_layers += [
            LinearCBP(feature_dim, hidden_dim, ema_decay=ema_decay),
            nn.ReLU(inplace=True),
        ]

        # add additional hidden layer for pixels
        if obs_type == "pixels":
            policy_layers += [
                LinearCBP(hidden_dim, hidden_dim, ema_decay=ema_decay),
                nn.ReLU(inplace=True),
            ]

        policy_layers += [LinearCBP(hidden_dim, action_dim, ema_decay=ema_decay)]

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
    def __init__(
        self,
        obs_type,
        obs_dim,
        action_dim,
        feature_dim,
        hidden_dim,
        normalize_basis_features_in_critic,
        ema_decay,
    ):
        super().__init__()

        self.obs_type = obs_type
        self.normalize_basis_features_in_critic = normalize_basis_features_in_critic

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.critic_trunk = nn.Sequential(
                LinearCBP(obs_dim, feature_dim, ema_decay=ema_decay),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )
            trunk_dim = feature_dim + action_dim

        else:
            # for states actions come in the beginning
            self.critic_trunk = nn.Sequential(
                LinearCBP(obs_dim + action_dim, hidden_dim, ema_decay=ema_decay),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                LinearCBP(trunk_dim, hidden_dim, ema_decay=ema_decay),
                nn.ReLU(inplace=True),
            ]

            if obs_type == "pixels":
                q_layers += [
                    LinearCBP(hidden_dim, hidden_dim, ema_decay=ema_decay),
                    nn.ReLU(inplace=True),
                ]

            q_layers += [LinearCBP(hidden_dim, 1, ema_decay=ema_decay)]

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


class TD3_continual_backprop_Agent(TD3Agent):
    def __init__(self, **kwargs: Unpack[TD3_continual_backprop_AgentKwargs]):

        self._kwargs = kwargs
        super().__init__(**kwargs)

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

        self.actor_cbp_layers = [n for n, m in self.actor.named_modules()
                            if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")]
        self.critic_cbp_layers = [n for n, m in self.critic.named_modules()
                             if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")]

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

        self.continual_backprop_refresh_mlp(
            model=self.critic,
            optimizer=self.critic_opt,
            out_map=self.CRITIC_OUT_MAP,
            replacement_rate=self._kwargs["replacement_rate"],
            bottom_frac=self._kwargs["bottom_frac"],
            maturity_threshold=self._kwargs["maturity_threshold"],
        )
        #
        # metrics = self.log_cbp_layer_metrics(
        #     metrics, model=self.critic, tracker=self.critic_tracker, prefix="critic"
        # )

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

        self.continual_backprop_refresh_mlp(
            model=self.actor,
            optimizer=self.actor_opt,
            out_map=self.ACTOR_OUT_MAP,
            replacement_rate=self._kwargs["replacement_rate"],
            bottom_frac=self._kwargs["bottom_frac"],
            maturity_threshold=self._kwargs["maturity_threshold"],
        )

        if self._kwargs["use_tb"] or self._kwargs["use_wandb"]:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        # metrics = self.log_cbp_layer_metrics(
        #     metrics, model=self.actor, tracker=self.actor_tracker, prefix="actor"
        # )

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self._kwargs["update_every_steps"] != 0:
            metrics = self.log_cbp_layer_metrics_fast(
                metrics,
                self.actor, self.actor_tracker,
                prefix="actor", step=step,
                layer_names_cache=self.actor_cbp_layers,
                rank_every=200,  # rank every 200 steps
                rank_iters=6,  # fewer iterations is faster
                rank_sample_units=256,  # smaller subsample
                rank_sample_batch=128,  # smaller subsample
            )

            metrics = self.log_cbp_layer_metrics_fast(
                metrics,
                self.critic, self.critic_tracker,
                prefix="critic", step=step,
                layer_names_cache=self.critic_cbp_layers,
                rank_every=200,
                rank_iters=6,
                rank_sample_units=256,
                rank_sample_batch=128,
            )
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
        all_params = (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        return sum(p.numel() for p in all_params)

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

    @torch.no_grad()
    def _zero_adam_slots_for_units(
        self,
        optimizer: torch.optim.Optimizer,
        param: torch.Tensor,
        unit_idxs: torch.Tensor,
        dim: int,
    ):
        """Zero Adam moments for selected unit slices of a param tensor."""
        st = optimizer.state.get(param, None)
        if st is None:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            buf = st.get(key, None)
            if buf is None:
                continue
            if dim == 0:
                buf[unit_idxs] = 0
            else:
                # move unit-dim to front, zero rows, move back
                perm = list(range(buf.ndim))
                perm[0], perm[dim] = perm[dim], perm[0]
                inv = [0] * buf.ndim
                for i, p in enumerate(perm):
                    inv[p] = i
                v = buf.permute(perm)
                v[unit_idxs] = 0
                buf.copy_(v.permute(inv))

    def _pick_bottom_tail_eligible(
        self,
        ema: torch.Tensor,
        age: torch.Tensor,
        k: int,
        maturity_threshold: int,
        bottom_frac: float,
    ) -> torch.Tensor:
        """Choose up to k indices among mature units from the bottom tail of utilities."""
        dev = ema.device
        eligible = age >= maturity_threshold
        if not eligible.any():
            return torch.empty(0, dtype=torch.long, device=dev)

        ema_elig = ema.clone()
        ema_elig[~eligible] = float("inf")

        out = ema.numel()
        tail = max(1, int(out * bottom_frac))
        tail_idxs = torch.argsort(ema_elig)[:tail]
        tail_idxs = tail_idxs[torch.isfinite(ema_elig[tail_idxs])]
        if tail_idxs.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=dev)

        k = min(k, tail_idxs.numel())
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=dev)
        perm = torch.randperm(tail_idxs.numel(), device=dev)[:k]
        return tail_idxs[perm]

    @torch.no_grad()
    def _reinit_linear_units(self, layer: nn.Linear, sel: torch.Tensor):
        """Reinitialize incoming weights/bias of selected output units; reset utility+age."""
        if sel.numel() == 0:
            return
        fan_in = layer.in_features
        std = 1.0 / (fan_in**0.5)  # LeCun normal-ish
        new_w = (
            torch.randn(
                (sel.numel(), layer.in_features),
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            * std
        )
        layer.weight[sel, :] = new_w
        if layer.bias is not None:
            layer.bias[sel] = 0.0
        # reset CBP buffers if present
        if hasattr(layer, "utility_ema"):
            layer.utility_ema[sel] = 0.0
        if hasattr(layer, "age"):
            layer.age[sel] = 0

    @torch.no_grad()
    def continual_backprop_refresh_mlp(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        replacement_rate: float = 1e-4,  # replacement rate
        k_per_layer: int = 1,
        bottom_frac: float = 0.1,
        maturity_threshold: int = 100,
        out_map: Dict[str, str],
        zero_adam: bool = False,
    ):
        """
        Paper-faithful CBP for MLPs:
          - increment ages,
          - pick mature lowest-utility units,
          - reinit incoming (this layer),
          - zero optimizer slots for those rows,
          - zero OUTGOING weights (next layer columns) + their Adam slots.
        out_map: dict mapping 'layer_name' -> 'next_layer_name' for zero-outgoing.
        """
        # 0) tick ages once
        for _, m in model.named_modules():
            if hasattr(m, "tick_age"):
                m.tick_age()

        named = dict(model.named_modules())

        # 1) per-layer: increment counters and refresh while >= 1
        for name, m in model.named_modules():
            # only CBP layers that we also map to consumers
            if name not in out_map:
                continue
            if not (
                hasattr(m, "utility_ema")
                and hasattr(m, "age")
                and hasattr(m, "cbp_counter")
            ):
                continue

            # increment counter by replacement rate * num_mature
            mature_mask = m.age >= maturity_threshold
            num_mature = int(mature_mask.sum().item())
            if num_mature > 0:
                m.cbp_counter.add_(replacement_rate * float(num_mature))

            # refresh while counter >= 1
            # one at a time (k=1) to mirror the algorithm
            while m.cbp_counter.item() >= 1.0:
                # pick ONE mature, low-utility unit from bottom tail
                sel = self._pick_bottom_tail_eligible(
                    m.utility_ema,
                    m.age,
                    k=k_per_layer,
                    maturity_threshold=maturity_threshold,
                    bottom_frac=bottom_frac,
                )

                if sel.numel() == 0:
                    # nothing eligible; stop to avoid infinite loop
                    break

                # reinit incoming (this layer)
                self._reinit_linear_units(m, sel)
                m.cbp_resets.add_(int(sel.numel()))
                if zero_adam:
                    self._zero_adam_slots_for_units(optimizer, m.weight, sel, dim=0)
                    if getattr(m, "bias", None) is not None:
                        self._zero_adam_slots_for_units(optimizer, m.bias, sel, dim=0)

                # zero outgoing in next layer(s)
                nxt_names = out_map[name]
                if isinstance(nxt_names, str):
                    nxt_names = [nxt_names]
                for nxt_name in nxt_names:
                    if nxt_name not in named:
                        raise KeyError(
                            f"{nxt_name!r} not found in model.named_modules()"
                        )
                    nxt = named[nxt_name]
                    if not hasattr(nxt, "weight"):
                        raise TypeError(f"{nxt_name!r} has no .weight")
                    nxt.weight[:, sel] = 0.0
                    if zero_adam:
                        self._zero_adam_slots_for_units(
                            optimizer, nxt.weight, sel, dim=1
                        )

                # decrement counter by 1 per refresh
                m.cbp_counter.sub_(1.0)

    def _stable_rank(self, x: torch.Tensor) -> float:
        """
        Effective rank = ||X||_F^2 / ||X||_2^2 on a (B,U) matrix of activations.
        """
        if x.numel() == 0:
            return float("nan")
        # center across batch (optional but often nicer)
        x = x - x.mean(dim=0, keepdim=True)
        # Frobenius norm squared
        fro2 = (x * x).sum().item()
        # spectral norm (largest singular value)
        # guard for small B,U
        s = torch.linalg.svdvals(x)
        spec2 = (s[0].item() ** 2) if s.numel() > 0 else float("nan")
        if spec2 == 0 or math.isnan(spec2):
            return float("nan")
        return fro2 / spec2

    def _get_cbp_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        return {
            name: m
            for name, m in model.named_modules()
            if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")
        }

    @torch.no_grad()
    def log_cbp_layer_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        model: nn.Module,
        tracker: "ActivationsTracker|None",
        prefix: str,
        dead_eps: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        Logs per-layer diagnostics to wandb for all CBP layers in `model`.
        """
        layers = self._get_cbp_layers(model)
        for name, m in layers.items():
            util = m.utility_ema.detach()
            age = m.age.detach().to(torch.float32)
            dead_frac = (util < dead_eps).float().mean().item()
            util_mean, util_median = util.mean().item(), util.median().item()
            age_mean = age.mean().item()

            # weight norms (L2), for kernel only
            w = m.weight.detach()
            w_l2 = w.norm(2).item()

            # replacement rate counter & resets if present
            counter = float(getattr(m, "cbp_counter", torch.tensor(0.0)).item())
            resets = int(getattr(m, "cbp_resets", torch.tensor(0)).item())

            # effective rank from tracker (if available)
            rank = None
            if tracker is not None and name in tracker.buffers:
                rank = self._stable_rank(tracker.buffers[name])

            metrics[f"{prefix}/{name}/dead_frac"] = dead_frac
            metrics[f"{prefix}/{name}/util_mean"] = util_mean
            metrics[f"{prefix}/{name}/util_median"] = util_median
            metrics[f"{prefix}/{name}/age_mean"] = age_mean
            metrics[f"{prefix}/{name}/w_l2"] = w_l2
            metrics[f"{prefix}/{name}/counter"] = counter
            metrics[f"{prefix}/{name}/resets_cum"] = resets

            if rank is not None:
                metrics[f"{prefix}/{name}/rank_eff"] = rank

        return metrics

    @torch.no_grad()
    def _power_iteration_top_sv(self, X: torch.Tensor, iters: int = 10) -> float:
        """
        Approximate largest singular value via power iteration.
        X: (B, U) activations (float32)
        """
        if X.numel() == 0:
            return float("nan")
        # center by batch for stability (optional)
        X = X - X.mean(dim=0, keepdim=True)
        # v in R^U
        U = X.shape[1]
        v = torch.randn(U, device=X.device, dtype=X.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            # u ← X v; v ← Xᵀ u
            u = X @ v
            un = u.norm() + 1e-12
            u = u / un
            v = X.t() @ u
            vn = v.norm() + 1e-12
            v = v / vn
        # Rayleigh quotient gives σ_max
        sigma = (X @ v).norm().item()
        return sigma

    @torch.no_grad()
    def _effective_rank_fast(self, X: torch.Tensor, iters: int = 8) -> float:
        """
        Stable rank ≈ ||X||_F^2 / ||X||_2^2 using power-iteration for ||X||_2.
        """
        if X.numel() == 0:
            return float("nan")
        X = X.to(dtype=torch.float32)  # keep math stable
        fro2 = (X * X).sum().item()
        sigma_max = self._power_iteration_top_sv(X, iters=iters)
        if sigma_max <= 0 or math.isnan(sigma_max):
            return float("nan")
        return fro2 / (sigma_max * sigma_max)

    def _get_cbp_layers_cached(self, model: nn.Module, cache: Optional[List[str]] = None) -> List[str]:
        if cache is not None:
            return cache
        # cache layer names with CBP buffers
        return [n for n, m in model.named_modules()
                if hasattr(m, "utility_ema") and hasattr(m, "age") and hasattr(m, "weight")]

    @torch.no_grad()
    def log_cbp_layer_metrics_fast(
            self,
            metrics: Dict[str, torch.Tensor],
            model: nn.Module,
            tracker,  # ActivationsTracker or None
            *,
            prefix: str,
            step: int,
            layer_names_cache: Optional[List[str]] = None,
            dead_eps: float = 1e-3,
            # performance knobs:
            rank_every: int = 200,  # compute rank every N steps (set 0 to disable)
            rank_iters: int = 8,  # power-iteration steps
            rank_sample_units: int = 512,  # subsample columns for rank
            rank_sample_batch: int = 256,  # subsample batch rows for rank
            hist_every: int = 1000,  # log histogram every N steps (set 0 to disable)
            hist_max_units: int = 512,  # cap units sent to W&B histogram
    ):
        """
        Faster W&B logging for CBP layers.
        - One wandb.log call (single dict).
        - Rank via power iteration, subsampled.
        - Histograms throttled & capped.
        """
        log_data = {}
        layer_names = self._get_cbp_layers_cached(model, layer_names_cache)
        named = dict(model.named_modules())

        for name in layer_names:
            m = named[name]
            util = m.utility_ema.detach()
            age = m.age.detach()

            # Scalars are cheap
            dead_frac = (util < dead_eps).float().mean().item()
            util_mean = util.mean().item()
            util_med = util.median().item()
            age_mean = age.to(torch.float32).mean().item()
            w_l2 = m.weight.detach().norm(2).item()
            counter = float(getattr(m, "cbp_counter", torch.tensor(0.0)).item())
            resets = int(getattr(m, "cbp_resets", torch.tensor(0)).item())

            base = f"{prefix}/{name}"
            metrics[f"{base}/dead_frac"] = dead_frac
            metrics[f"{base}/util_mean"] = util_mean
            metrics[f"{base}/util_median"] = util_med
            metrics[f"{base}/age_mean"] = age_mean
            metrics[f"{base}/w_l2"] = w_l2
            metrics[f"{base}/counter"] = counter
            metrics[f"{base}/resets_cum"] = resets

            # Rank (throttled + subsampled)
            if rank_every and (step % rank_every == 0) and tracker is not None and name in tracker.buffers:
                X = tracker.buffers[name]  # (B, U)
                # Subsample along batch and units to keep it fast
                if X.size(0) > rank_sample_batch:
                    idx_b = torch.randint(X.size(0), (rank_sample_batch,), device=X.device)
                    X = X.index_select(0, idx_b)
                if X.size(1) > rank_sample_units:
                    idx_u = torch.randint(X.size(1), (rank_sample_units,), device=X.device)
                    X = X.index_select(1, idx_u)
                rank_eff = self._effective_rank_fast(X, iters=rank_iters)
                metrics[f"{base}/rank_eff"] = rank_eff

        return metrics
