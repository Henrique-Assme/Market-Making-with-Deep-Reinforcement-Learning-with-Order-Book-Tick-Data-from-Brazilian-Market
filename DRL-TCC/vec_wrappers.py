from __future__ import annotations

from stable_baselines3.common.vec_env import VecNormalize


class LoggingVecNormalize(VecNormalize):
    """VecNormalize wrapper that also forwards normalized rewards to the underlying env logs."""

    def __init__(self, venv, *, log_method: str = "record_normalized_reward", **kwargs):
        super().__init__(venv, **kwargs)
        self._normalized_reward_log_method = log_method

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        method_name = self._normalized_reward_log_method
        if method_name and hasattr(self.venv, "env_method"):
            for idx, rew in enumerate(rewards):
                try:
                    self.venv.env_method(method_name, float(rew), indices=[idx])
                except (AttributeError, NotImplementedError):
                    method_name = None
                    break
        for idx, info in enumerate(infos):
            if isinstance(info, dict) and "normalized_reward" not in info:
                info["normalized_reward"] = float(rewards[idx])
        return obs, rewards, dones, infos
        