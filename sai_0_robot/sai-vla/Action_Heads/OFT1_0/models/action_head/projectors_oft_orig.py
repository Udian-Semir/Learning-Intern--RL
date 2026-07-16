"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module): # ! Used
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class ProprioProjector_Changed(nn.Module): # ! Added and Changed
    """
    Projects proprio state inputs into the LLM's embedding space with specific padding and normalization.
    e.g. Converts (bsz, 16) -> (bsz, llm_dim) by padding zeros at front and placing original values at end,
         then normalizes and expands to (bsz, 1, llm_dim).
    """
    def __init__(self, llm_dim: int, proprio_dim: int = 16) -> None:
        super().__init__()
        self.llm_dim = llm_dim 
        self.proprio_dim = proprio_dim 
        self.layer_norm = nn.LayerNorm(self.llm_dim)

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        bsz = proprio.shape[0]
        projected_features = torch.zeros(bsz, self.llm_dim, device=proprio.device, dtype=proprio.dtype)
        projected_features[:, -self.proprio_dim:] = proprio
        # projected_features = self.layer_norm(projected_features) # ! 因为进入动作head前会有LayerNorm，所以这里不需要再norm一次
        
        # Expand dimension to (bsz, 1, llm_dim)
        projected_features = projected_features.unsqueeze(1)
        
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
