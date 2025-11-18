from stable_baselines3.common.callbacks import BaseCallback

class ActionLoggerCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            
        def _on_step(self):
            return super()._on_step()

        def _on_rollout_end(self) -> None:
            try:
                # actions shape: (n_steps, n_envs, 1) or (n_steps, n_envs)
                acts = self.model.rollout_buffer.actions
                import numpy as np
                a = acts
                if hasattr(a, "cpu"):
                    a = a.cpu().numpy()
                a = np.asarray(a)
                a = a.reshape(-1)
                total = max(len(a), 1)
                # count each action id 0..3
                counts = {i: int((a == i).sum()) for i in range(4)}
                fracs = {i: counts[i] / total for i in range(4)}
                # record to TB
                self.logger.record("train/action_frac_bid", fracs.get(0, 0.0))
                self.logger.record("train/action_frac_ask", fracs.get(1, 0.0))
                self.logger.record("train/action_frac_both", fracs.get(2, 0.0))
                self.logger.record("train/action_frac_no_quote", fracs.get(3, 0.0))
                # also print a compact summary
                # print(f"[ActionLogger] counts={counts} fracs={fracs}")
            except Exception as e:
                print(f"[ActionLogger] failed to log actions: {e}")