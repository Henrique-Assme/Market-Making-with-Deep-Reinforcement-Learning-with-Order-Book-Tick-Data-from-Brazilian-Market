from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from backbone import Backbone

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=(4,64)):
        super().__init__(observation_space, features_dim=hidden_dim[1])
        input_size = observation_space.shape[0]
        self.backbone = Backbone(input_size, hidden_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_dim=(4,64), **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs={'hidden_dim': hidden_dim}
        )