import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn

import wandb
from dm_env import specs

import dmc
import utils
import uuid
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics
# from shimmy import DmControlCompatibilityV0


import gymnasium as gym

from video import TrainVideoRecorder, VideoRecorder
from collections import OrderedDict

from modulators import NoisySineModulator

torch.backends.cudnn.benchmark = True
from absl import logging

from dmc_benchmark import (
    PRIMAL_TASKS,
    PRIMAL_TASKS_WALK,
    PRIMAL_TASKS_FAST_RUN,
    PRIMAL_TASKS_RUN_BACKWARD,
    CRL_TASKS_SAME_REWARD,
    CRL_TASKS_DIFF_REWARD,
    CRL_DIFF_DYNAMICS_DIFF_REWARD,
    CRL_TASKS_DIFF_RUN_SPEED_REWARD,
    CRL_DIFF_DOMAINS_SAME_REWARD,
    CRL_WALKER_WALK_RUN_TASKS,
    CRL_WALKER_STAND_RUN_TASKS,
    CRL_RUN_JUMP_TASKS,
    CRL_DIFF_DOMAINS_DIFF_REWARD,
    CRL_DIFF_DOMAINS_DIFF_REWARD_CHEETAH_FISH,
)

num_actions = {
    "cheetah": 6,
    "walker": 6,
    "quadruped": 12,
    "cartpole": 1,
    "humanoid": 21,
    "fish": 5,
    "dog": 38,
    "finger": 2,
}


def make_agent(obs_type, obs_spec, action_spec, domain, batch_size, minibatch_size, num_iterations, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.domain = domain
    cfg.batch_size = batch_size
    cfg.minibatch_size = minibatch_size
    cfg.num_iterations = num_iterations
    logging.info("agent config: %s", cfg)
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):

        if cfg.mila_env:
            self.work_dir = Path(cfg.mila_work_dir)

        else:
            self.work_dir = Path(cfg.work_dir)

        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # create envs
        if self.cfg.single_task_run:
            logging.info("Single task run")
            self.tasks = [PRIMAL_TASKS[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        elif self.cfg.single_task_walk:
            logging.info("Single task walk")
            self.tasks = [PRIMAL_TASKS_WALK[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        elif self.cfg.single_task_run_fast:
            logging.info("Single task run fast")
            self.tasks = [PRIMAL_TASKS_FAST_RUN[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        elif self.cfg.single_task_run_backward:
            logging.info("Single task run backward")
            self.tasks = [PRIMAL_TASKS_RUN_BACKWARD[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        # walk run tasks (only for walker domain)
        elif self.cfg.walk_run_tasks:
            assert self.cfg.domain == "walker", "Walk run tasks only for walker domain"
            logging.info("Walk run tasks for walker")

            self.tasks = CRL_WALKER_WALK_RUN_TASKS[self.cfg.domain]

        # stand run tasks (only for walker domain)
        elif self.cfg.stand_run_tasks:
            assert self.cfg.domain == "walker", "Stand run tasks only for walker domain"
            logging.info("Stand run tasks for walker")

            self.tasks = CRL_WALKER_STAND_RUN_TASKS[self.cfg.domain]

        # run jump tasks (only for quadruped domain)
        elif self.cfg.run_jump_tasks:
            assert (
                self.cfg.domain == "quadruped"
            ), "Run jump tasks only for quadruped domain"
            logging.info("Run jump tasks for quadruped")

            self.tasks = CRL_RUN_JUMP_TASKS[self.cfg.domain]

        # different dynamics but same reward function for all tasks
        elif (
            self.cfg.diff_dynamics_for_all_tasks and self.cfg.same_reward_for_all_tasks
        ):
            logging.info("Different dynamics but same reward function for all tasks")
            self.tasks = CRL_TASKS_SAME_REWARD[self.cfg.domain]

        # different dynamics and different reward function for all tasks
        elif (
            self.cfg.diff_dynamics_for_all_tasks
            and not self.cfg.same_reward_for_all_tasks
        ):
            logging.info(
                "Different dynamics and different reward function for all tasks"
            )
            self.tasks = CRL_DIFF_DYNAMICS_DIFF_REWARD[self.cfg.domain]

        # different run speed but same reward function for all tasks
        elif self.cfg.diff_run_speed_for_all_tasks:
            logging.info("Different run speed but same reward function for all tasks")
            self.tasks = CRL_TASKS_DIFF_RUN_SPEED_REWARD[self.cfg.domain]

        # different domains (eg. cheetah-run and walker-run) with same reward function for all tasks
        elif self.cfg.diff_domains_same_reward:
            logging.info("Different domains with same reward function for all tasks")
            self.tasks = CRL_DIFF_DOMAINS_SAME_REWARD

        # different domains and different reward function (eg. cheetah-run and quadruped-jump)
        elif self.cfg.diff_domains_diff_reward:
            logging.info(
                "Different domains with different reward function for all tasks"
            )
            self.tasks = CRL_DIFF_DOMAINS_DIFF_REWARD

        elif self.cfg.diff_domains_diff_reward_cheetah_fish:
            logging.info("Different domains with different task for cheetah and fish")
            self.tasks = CRL_DIFF_DOMAINS_DIFF_REWARD_CHEETAH_FISH

        # same dynamics but different reward function for all tasks (eg. run forward and backward)
        else:
            assert (
                not self.cfg.diff_dynamics_for_all_tasks
                and not self.cfg.same_reward_for_all_tasks
            ), "Invalid tasks configuration"
            logging.info("Same dynamics but different reward function for all tasks")
            self.tasks = CRL_TASKS_DIFF_REWARD[self.cfg.domain]

        self.num_tasks = len(self.tasks)
        self._current_task_id = 0  # task id always starts from 0

        # create video recorders
        # self.eval_video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_eval_video else None,
        #     camera_id=0 if "quadruped" not in self.cfg.domain else 2,
        #     use_wandb=self.cfg.use_wandb,
        # )

        # self.train_video_recorder = TrainVideoRecorder(
        #     self.work_dir if cfg.save_train_video else None,
        #     camera_id=0 if "quadruped" not in self.cfg.domain else 2,
        #     use_wandb=self.cfg.use_wandb,
        # )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._exposure_id = 0
        self._update_step = 0

        self.train_envs = []
        self.eval_envs = []
        self.obs_specs = []
        self.num_actions = []

        name = self.tasks[0]
        domain, task = name.split("_", 1)
        self.valid_actions = num_actions[domain]

        self.env_rng = np.random.default_rng(seed=self.cfg.env_seed)
        mass_scale = self.env_rng.uniform(0.95, 1.05)
        friction_scale = self.env_rng.uniform(0.9, 1.1)

        # create new training and eval environment
        train_env = dmc.make(
            name,
            self.cfg.obs_type,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            num_actions=self.valid_actions,
            num_valid_actions=self.valid_actions,
            normalize_observation=self.cfg.normalize_observation,
            device=self.cfg.device,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
        )

        logging.info("task: %s", self.tasks[0])

        # get the max shape of obs_spec and action_spec
        obs_spec = specs.Array(
            shape=train_env.observation_spec().shape,
            dtype=np.float32,
            name="observation",
        )

        action_spec = specs.Array(
            shape=np.array([self.valid_actions]),
            dtype=np.float32,
            name="action",
        )

        self.batch_size = int(self.cfg.num_envs * self.cfg.agent.num_steps)
        self.minibatch_size = int(self.batch_size // self.cfg.agent.num_minibatches)
        self.num_iterations = self.cfg.num_train_frames // self.batch_size

        print("batch_size: ", self.batch_size)
        print("minibatch_size: ", self.minibatch_size)
        print("num_iterations: ", self.num_iterations)

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            obs_spec,
            action_spec,
            cfg.domain,
            self.batch_size,
            self.minibatch_size,
            self.num_iterations,
            cfg.agent,
        )

        logging.info("num of parameters: %s", self.agent.num_params())

        logging.info("agent: %s", self.agent.__class__.__name__)

        # if self.cfg.agent.consolidation:
        #     self.agent.update_storage_timescale_check(total_num_updates)
        #     self.agent.first_task_update_storage_timescale_check(num_updates_per_task)

        self.plasticity_injection_step = int(
            self.cfg.num_train_frames * self.cfg.plasticity_injection
        )

        # generate a uuid for the replay directory
        self.uuid = uuid.uuid4().hex

        # # add only uuid to replay directory
        # replay_dir = self.work_dir / "buffer" / self.uuid
        #
        # # log the replay directory
        # logging.info("replay_dir: %s", replay_dir)
        #
        # # create data storage
        # self.replay_storage = ReplayBufferStorage(
        #     data_specs, meta_specs, replay_dir, self.cfg.env_type
        # )
        #
        # # create replay buffer
        # self.replay_loader = make_replay_loader(
        #     self.replay_storage,
        #     cfg.replay_buffer_size,
        #     cfg.batch_size,
        #     cfg.replay_buffer_num_workers,
        #     False,
        #     cfg.nstep,
        #     cfg.discount,
        # )
        #
        # self._replay_iter = None

        # flatten the cfg file
        self._cfg_flatten = utils.dictionary_flatten(self.cfg)

        logging.info("{}\n".format(self._cfg_flatten))

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.agent.name,
                    cfg.domain,
                    cfg.obs_type,
                    str(cfg.seed),
                ]
            )

            # get current working directory and add wandb_dir
            wandb_dir_absolute = Path.cwd()

            # convert wandb_dir_absolute to string
            wandb_dir_str = wandb_dir_absolute.as_posix()

            # log wandb_dir_str
            logging.info("wandb_dir_str: %s", wandb_dir_str)

            project_name = "continual_rl" + self.cfg.domain + "_" + self.cfg.mode
            wandb.init(
                project=project_name,
                group=cfg.agent.name,
                name=exp_name,
                config=self._cfg_flatten,
                dir=wandb_dir_str,
                mode=self.cfg.wandb_mode,
                settings=wandb.Settings(
                    start_method="thread"
                ),  # required for offline mode
            )

        else:
            wandb.init(mode="disabled")

        if self.agent.name == "sf_simple_beakers_continuous_attention":

            for j in range(self.agent.sf_dim + 1):
                wandb.log({f"timescale_dim": j})

            timescale_embedding = self.agent.timescale_embedding
            if timescale_embedding is not None:
                # save timescale embedding as a csv file to workdir
                timescale_embedding_csv = self.work_dir / "timescale_embedding.csv"
                np.savetxt(
                    timescale_embedding_csv,
                    timescale_embedding.cpu().numpy(),
                    delimiter=",",
                )
                logging.info("timescale_embedding saved to %s", timescale_embedding_csv)

        # Create modulators with different seeds
        self.mass_mod = NoisySineModulator(
            period=self.cfg.mass_modulator_period / int(self.cfg.num_envs /4),
            phase=0.0,
            seed=1,
            min_val=1 - self.cfg.percentage_shifts_mass,
            max_val=1 + self.cfg.percentage_shifts_mass,
        )

        self.friction_mod = NoisySineModulator(
            period=self.cfg.friction_modulator_period,
            phase=0.0,
            seed=2,
            min_val=self.cfg.friction_min_val,
            max_val=self.cfg.friction_max_val,
        )

        self.forward_backwards_mod = NoisySineModulator(
            period=self.cfg.forward_backwards_modulator_period,
            phase=0.0,
            seed=3,
            min_val=self.cfg.forward_backwards_min_val,
            max_val=self.cfg.forward_backwards_max_val,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def update_step(self):
        return self._update_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(
        self,
        current_eval_env,
        valid_actions: int,
        name: str,
        meta=None,
        mass_scale: float = 1.0,
        friction_scale: float = 1.0,
        forward_scale: float = 1.0,
        use_forward_scale: bool = False,
    ):

        assert meta is not None, "meta must be provided for evaluation"

        obs, _ = current_eval_env.reset()
        episodic_returns = []
        episode_lengths = []
        while len(episodic_returns) < self.cfg.num_eval_episodes:
            actions, _, _, _ = self.agent.get_action_and_value(torch.Tensor(obs).to(self.cfg.device))
            next_obs, _, _, _, infos = current_eval_env.step(actions.cpu().numpy())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episode_length={info['episode']['l']}")
                    episodic_returns += [info["episode"]["r"]]
                    episode_lengths += [info["episode"]["l"]]
            obs = next_obs

        # return episodic_returns
        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            log("episode_reward", np.mean(episodic_returns))
            log("episode_length", np.mean(episode_lengths))
            log("episode", self.global_episode)
            log("step", self.global_step)
            log("task_id", self._current_task_id)
            log("exposure_id", self._exposure_id)
            log("mass", mass_scale)
            log("friction", friction_scale)
            log("forward_scale", forward_scale)



    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        total_returns = 0

        if self.agent.consolidation:
            self.agent.num_train_frames = (
                self.cfg.num_train_frames // self.cfg.action_repeat
            )
            self.agent.set_up_consolidation_system()

        for exposure_id in range(self.cfg.num_exposures):
            if self.cfg.terminate_after_first_task and exposure_id > 0:
                break
            for task_id in range(self.num_tasks):

                total_returns_task = 0
                if self.cfg.terminate_after_first_task and task_id > 0:
                    break
                task_step = 0
                self._current_task_id = task_id
                self._exposure_id = exposure_id

                mass_scale = self.mass_mod.sample(self._global_step)
                friction_scale = self.friction_mod.sample(self._global_step)
                forward_scale = self.forward_backwards_mod.sample(self._global_step)

                name = self.tasks[self._current_task_id]
                domain, task = name.split("_", 1)

                logging.info("current domain: %s", domain)
                logging.info("current task: %s", task)

                current_train_env = SyncVectorEnv([
                    lambda: dmc.make_dm_compatible(
                        name,
                        self.cfg.obs_type,
                        self.cfg.frame_stack,
                        self.cfg.action_repeat,
                        self.cfg.seed,
                        num_actions=self.valid_actions,
                        num_valid_actions=self.valid_actions,
                        normalize_observation=self.cfg.normalize_observation,
                        device=self.cfg.device,
                        mass_scale=mass_scale,
                        friction_scale=friction_scale,
                        forward_scale=forward_scale,
                        use_forward_scale=self.cfg.use_forward_scale,
                    )
                    for _ in range(self.cfg.num_envs)
                ])

                current_train_env = RecordEpisodeStatistics(current_train_env)
                current_train_env = gym.wrappers.vector.NormalizeObservation(current_train_env)
                current_train_env = gym.wrappers.vector.TransformObservation(
                    current_train_env, lambda obs: np.clip(obs, -10, 10), current_train_env.single_observation_space
                )
                # current_train_env = gym.wrappers.vector.NormalizeReward(current_train_env, gamma=self.cfg.discount)
                # current_train_env = gym.wrappers.vector.TransformReward(
                #     current_train_env, lambda r: np.clip(r, -10, 10)
                # )

                obs = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs) + current_train_env.single_observation_space.shape).to(self.cfg.device)
                actions = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs) + current_train_env.single_action_space.shape).to(self.cfg.device)
                logprobs = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                rewards = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                dones = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                values = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)

                elapsed_time, total_time = self.timer.reset()
                next_obs, temp = current_train_env.reset(seed=self.cfg.env_seed)
                next_obs = torch.Tensor(next_obs).to(self.cfg.device)
                next_done = torch.zeros(self.cfg.num_envs).to(self.cfg.device)

                episode_step = 0
                # episode_discount = self.cfg.discount
                # time_step = current_train_env.reset()
                #
                meta = self.agent.init_meta()
                # # self.replay_storage.add(time_step, meta)
                # # self.train_video_recorder.init(time_step.observation)
                metrics = {}
                for iteration in range(1, self.num_iterations + 1):

                    # Annealing the rate if instructed to do so.
                    if self.cfg.agent.anneal_lr:
                        frac = 1.0 - (iteration - 1.0) / self.num_iterations
                        lrnow = frac * self.cfg.agent.lr
                        self.agent.optimizer.param_groups[0]["lr"] = lrnow

                    for step in range(0, self.cfg.agent.num_steps):
                        self._global_step += self.cfg.num_envs
                        obs[step] = next_obs
                        dones[step] = next_done

                        # ALGO LOGIC: action logic
                        with torch.no_grad():
                            action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                            values[step] = value.flatten()

                        actions[step] = action
                        logprobs[step] = logprob

                        next_obs, reward, terminations, truncations, infos = current_train_env.step(action.cpu().numpy())
                        next_done = np.logical_or(terminations, truncations)
                        rewards[step] = torch.tensor(reward).to(self.cfg.device).view(-1)
                        next_obs, next_done = torch.Tensor(next_obs).to(self.cfg.device), torch.Tensor(next_done).to(self.cfg.device)

                        episode_step += 1
                        task_step += 1

                        if "episode" in infos and "_episode" in infos:
                            ended_ids = np.where(infos["_episode"])[0]

                            if len(ended_ids) > 0:
                                elapsed_time, total_time = self.timer.reset()

                                ep_returns = infos["episode"]["r"][ended_ids]
                                ep_lengths = infos["episode"]["l"][ended_ids]

                                mean_return = ep_returns.mean()
                                mean_length = ep_lengths.mean()

                                metrics["episode_reward"] = mean_return
                                metrics["episode_length"] = mean_length
                                metrics["step"] = self.global_step
                                metrics["update_step"] = self.update_step
                                metrics["total_return"] = total_returns
                                metrics["total_return_task"] = total_returns_task
                                metrics["total_time"] = total_time
                                metrics["mass"] = mass_scale
                                metrics["friction"] = friction_scale
                                metrics["forward_scale"] = forward_scale
                                metrics["num_ended"] = len(ended_ids)

                                if self.global_episode % (self.cfg.log_freq) == 0:
                                    with self.logger.log_and_dump_ctx(
                                            self.global_frame, ty="train"
                                    ) as log:
                                        log("total_time", total_time)
                                        log("episode_reward", mean_return)
                                        log("episode_length", mean_length)
                                        log("episode", self.global_episode)
                                        log("step", self.global_step)
                                        log("task_id", task_id)
                                        log("total_returns", total_returns)
                                        log("total_returns_task", total_returns_task)
                                        log("exposure_id", exposure_id)
                                        log("mass", mass_scale)
                                        log("friction", friction_scale)
                                        log("forward_scale", forward_scale)
                                        log("update_step", self.update_step)

                                if (
                                        self.global_episode % self.cfg.switch_mass_friction_every_n_episodes == 0
                                ):

                                    mass_scale = self.mass_mod.sample(self._update_step)
                                    friction_scale = self.friction_mod.sample(self._update_step)
                                    forward_scale = self.forward_backwards_mod.sample(self._update_step)

                                self._global_episode += len(ended_ids)

                                current_train_env = SyncVectorEnv([
                                    lambda: dmc.make_dm_compatible(
                                        name,
                                        self.cfg.obs_type,
                                        self.cfg.frame_stack,
                                        self.cfg.action_repeat,
                                        self.cfg.seed,
                                        num_actions=self.valid_actions,
                                        num_valid_actions=self.valid_actions,
                                        normalize_observation=self.cfg.normalize_observation,
                                        device=self.cfg.device,
                                        mass_scale=mass_scale,
                                        friction_scale=friction_scale,
                                        forward_scale=forward_scale,
                                        use_forward_scale=self.cfg.use_forward_scale,
                                    )
                                    for _ in range(self.cfg.num_envs)
                                ])

                                current_train_env = RecordEpisodeStatistics(current_train_env)
                                current_train_env = gym.wrappers.vector.NormalizeObservation(current_train_env)
                                current_train_env = gym.wrappers.vector.TransformObservation(
                                    current_train_env, lambda obs: np.clip(obs, -10, 10),
                                    current_train_env.single_observation_space
                                )
                                # current_train_env = gym.wrappers.vector.NormalizeReward(current_train_env,
                                #                                                         gamma=self.cfg.discount)
                                # current_train_env = gym.wrappers.vector.TransformReward(
                                #     current_train_env, lambda r: np.clip(r, -10, 10)
                                # )

                                obs = torch.zeros((self.cfg.agent.num_steps,
                                                   self.cfg.num_envs) + current_train_env.single_observation_space.shape).to(
                                    self.cfg.device)
                                actions = torch.zeros((self.cfg.agent.num_steps,
                                                       self.cfg.num_envs) + current_train_env.single_action_space.shape).to(
                                    self.cfg.device)
                                logprobs = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                                rewards = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                                dones = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)
                                values = torch.zeros((self.cfg.agent.num_steps, self.cfg.num_envs)).to(self.device)

                                elapsed_time, total_time = self.timer.reset()
                                next_obs, _ = current_train_env.reset(seed=self.cfg.env_seed)
                                next_obs = torch.Tensor(next_obs).to(self.cfg.device)
                                next_done = torch.zeros(self.cfg.num_envs).to(self.cfg.device)


                    # bootstrap value if not done
                    with torch.no_grad():
                        next_value = self.agent.get_value(next_obs).reshape(1, -1)
                        advantages = torch.zeros_like(rewards).to(self.cfg.device)
                        lastgaelam = 0
                        for t in reversed(range(self.cfg.agent.num_steps)):
                            if t == self.cfg.agent.num_steps - 1:
                                nextnonterminal = 1.0 - next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                nextvalues = values[t + 1]
                            delta = rewards[t] + self.cfg.discount * nextvalues * nextnonterminal - values[t]
                            advantages[
                                t] = lastgaelam = delta + self.cfg.discount * self.cfg.agent.gae_lambda * nextnonterminal * lastgaelam
                        returns = advantages + values

                    # flatten the batch
                    b_obs = obs.reshape((-1,) + current_train_env.single_observation_space.shape)
                    b_logprobs = logprobs.reshape(-1)
                    b_actions = actions.reshape((-1,) + current_train_env.single_action_space.shape)
                    b_advantages = advantages.reshape(-1)
                    b_returns = returns.reshape(-1)
                    b_values = values.reshape(-1)

                    # Optimizing the policy and value network
                    b_inds = np.arange(self.cfg.agent.batch_size)
                    clipfracs = []

                    # try to evaluate
                    # if eval_every_step(self.global_step):
                    #     self.logger.log(
                    #         "eval_total_time",
                    #         self.timer.total_time(),
                    #         self.global_frame,
                    #     )
                    #     self.eval(
                    #         current_eval_env=current_train_env,
                    #         valid_actions=self.valid_actions,
                    #         meta=meta,
                    #         mass_scale=mass_scale,
                    #         friction_scale=friction_scale,
                    #         forward_scale=forward_scale,
                    #         use_forward_scale=self.cfg.use_forward_scale,
                    #         name=name,
                    #     )

                    for epoch in range(self.cfg.agent.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, self.cfg.agent.batch_size, self.cfg.agent.minibatch_size):
                            end = start + self.cfg.agent.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                          b_actions[mb_inds])
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > self.cfg.agent.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if self.cfg.agent.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.agent.clip_coef, 1 + self.cfg.agent.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if self.cfg.agent.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -self.cfg.agent.clip_coef,
                                    self.cfg.agent.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - self.cfg.agent.ent_coef * entropy_loss + v_loss * self.cfg.agent.vf_coef

                            self.agent.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.agent.params, self.cfg.agent.max_grad_norm)
                            self.agent.optimizer.step()
                            self._update_step += 1

                    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    metrics["learning_rate"] = self.agent.optimizer.param_groups[0]["lr"]
                    metrics["value_loss"] = v_loss.item()
                    metrics["policy_loss"] = pg_loss.item()
                    metrics["entropy"] = entropy_loss.item()
                    metrics["old_approx_kl"] = old_approx_kl.item()
                    metrics["approx_kl"] = approx_kl.item()
                    metrics["clipfrac"] = np.mean(clipfracs)
                    metrics["explained_variance"] = explained_var

                    self.logger.log_metrics(metrics, self.global_step, ty="train")

                # save snapshot at the end of each task
                if self.cfg.save_snapshot_after_each_task:
                    self.save_snapshot()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)

        if self.agent.consolidation:
            snapshot_dir = (
                snapshot_dir / f"beaker_capacity_{self.cfg.agent.beaker_capacity}"
            )
            snapshot_dir = snapshot_dir / f"init_tube_{self.cfg.agent.init_tube}"
            snapshot_dir = snapshot_dir / f"num_beakers_{self.cfg.agent.num_beakers}"
            snapshot_dir = (
                snapshot_dir
                / f"max_grad_norm_consolidation{self.cfg.agent.max_grad_norm_consolidation}"
            )

        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = [
            "agent",
            "_global_step",
            "_global_episode",
            "_exposure_id",
            "_current_task_id",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)
            logging.info(f"snapshot saved to {snapshot}")

    def load_snapshot(self, snapshot_path):
        with snapshot_path.open("rb") as f:
            payload = torch.load(f)
            for k, v in payload.items():
                self.__dict__[k] = v
            logging.info(f"snapshot loaded from {snapshot_path}")


@hydra.main(config_path=".", config_name="full_train_multitimescale", version_base=None)
def main(cfg):
    from full_train_multitimescale_ppo import Workspace as W

    workspace = W(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
