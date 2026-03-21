"""DreamerV3 trainer.

Reuses the existing Dreamer world-model implementation, but loads a dedicated
DreamerV3 config so it can be trained and benchmarked as a separate algorithm.
"""

from __future__ import annotations

import os

from algorithms.common import load_algo_config, save_run_config
from algorithms.dreamer.trainer import DreamerTrainer


class DreamerV3Trainer(DreamerTrainer):
    name = "DreamerV3"

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("dreamerv3", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 200_000))
        seed = overrides.get("seed", t.get("seed", 42))
        self.save_path = save_path or self.save_path.replace("dreamer", "dreamerv3")
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        tmp = self._build_probe_env(seed, game_ov, reward_cfg)
        self._obs_dim = tmp.observation_space.shape[0]
        self._act_dim = tmp.action_space.n
        tmp.close()

        rssm_h = hp.get("rssm_hidden", 256)
        rssm_s = hp.get("rssm_stochastic", 32)
        rssm_c = hp.get("rssm_discrete_classes", 32)

        self.rssm = self._build_rssm(rssm_h, rssm_s, rssm_c)
        feat_dim = self.rssm.feature_dim

        self.reward_pred = self._build_reward_predictor(feat_dim)
        self.continue_pred = self._build_continue_predictor(feat_dim)
        self.obs_decoder = self._build_obs_decoder(feat_dim)
        self.actor = self._build_actor(feat_dim)
        self.critic = self._build_critic(feat_dim)

        self._world_opt = self._build_world_optimizer(hp)
        self._actor_opt = self._build_actor_optimizer(hp)
        self._critic_opt = self._build_critic_optimizer(hp)

        self._hp = hp
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed

    def _build_probe_env(self, seed: int, game_ov: dict, reward_cfg: dict):
        from algorithms.common import make_env

        return make_env(0, seed, game_ov, reward_cfg)()

    def _build_rssm(self, rssm_h: int, rssm_s: int, rssm_c: int):
        from algorithms.dreamer.networks import RSSM

        return RSSM(self._obs_dim, self._act_dim, rssm_h, rssm_s, rssm_c).to(self.device)

    def _build_reward_predictor(self, feat_dim: int):
        from algorithms.dreamer.networks import RewardPredictor

        return RewardPredictor(feat_dim).to(self.device)

    def _build_continue_predictor(self, feat_dim: int):
        from algorithms.dreamer.networks import ContinuePredictor

        return ContinuePredictor(feat_dim).to(self.device)

    def _build_obs_decoder(self, feat_dim: int):
        from algorithms.dreamer.networks import ObsDecoder

        return ObsDecoder(feat_dim, self._obs_dim).to(self.device)

    def _build_actor(self, feat_dim: int):
        from algorithms.dreamer.networks import Actor

        return Actor(feat_dim, self._act_dim).to(self.device)

    def _build_critic(self, feat_dim: int):
        from algorithms.dreamer.networks import Critic

        return Critic(feat_dim).to(self.device)

    def _build_world_optimizer(self, hp: dict):
        world_params = (
            list(self.rssm.parameters())
            + list(self.reward_pred.parameters())
            + list(self.continue_pred.parameters())
            + list(self.obs_decoder.parameters())
        )
        return self._make_optimizer(world_params, hp.get("learning_rate_world", 3e-4))

    def _build_actor_optimizer(self, hp: dict):
        return self._make_optimizer(self.actor.parameters(), hp.get("learning_rate_actor", 1e-4))

    def _build_critic_optimizer(self, hp: dict):
        return self._make_optimizer(self.critic.parameters(), hp.get("learning_rate_critic", 3e-4))

    def _make_optimizer(self, params, lr: float):
        import torch

        return torch.optim.Adam(params, lr=lr)
