"""
Fixed-size trajectory replay buffer for Online Decision Transformer training.
"""

import random
from collections import namedtuple
from typing import Dict, List, Optional

import torch

DTTransition = namedtuple(
    "DTTransition",
    [
        "state",  # [state_dim] tensor
        "action",  # [action_dim] tensor (squashed, in [0.1, 10.0])
        "reward",  # float
        "rtg",  # float (return-to-go, filled by add_episode)
        "timestep",  # int (phase index within episode, 0-based)
        "done",  # bool
        # For optional state encoder fine-tuning (can be None)
        "gaussian_features",  # [N, feat_dim] or None
        "context",  # [3] or None
        "iteration",  # int
        "max_iterations",  # int
    ],
)


class TrajectoryReplayBuffer:
    """
    Fixed-size FIFO replay buffer that stores complete episodes.

    Each episode is a list of DTTransition namedtuples. When the buffer
    exceeds ``max_episodes``, the oldest episode is evicted.
    """

    def __init__(self, max_episodes: int = 100, discount: float = 1.0):
        self.max_episodes = max_episodes
        self.discount = discount
        self.episodes: List[List[DTTransition]] = []
        self.episode_returns: List[float] = []

    # ── Storage ────────────────────────────────────────────────────────────────

    def add_episode(self, transitions: List[DTTransition]) -> None:
        """
        Compute RTGs for each transition and store the episode.

        RTG is computed backwards:  rtg_T = r_T,  rtg_t = r_t + γ * rtg_{t+1}
        """
        if not transitions:
            return

        T = len(transitions)
        rtgs = [0.0] * T
        rtgs[-1] = transitions[-1].reward
        for t in reversed(range(T - 1)):
            rtgs[t] = transitions[t].reward + self.discount * rtgs[t + 1]

        episode = [t._replace(rtg=rtgs[i]) for i, t in enumerate(transitions)]
        episode_return = sum(t.reward for t in transitions)

        self.episodes.append(episode)
        self.episode_returns.append(episode_return)

        # FIFO eviction
        while len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
            self.episode_returns.pop(0)

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample_batch(
        self, batch_size: int, max_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample ``batch_size`` episodes uniformly and pad/crop to ``max_len``.

        Returns dict with keys:
            states:         [batch, max_len, state_dim]
            actions:        [batch, max_len, action_dim]
            rewards:        [batch, max_len]
            rtgs:           [batch, max_len, 1]
            timesteps:      [batch, max_len]  (LongTensor)
            attention_mask: [batch, max_len]  (BoolTensor, True=valid)
        """
        indices = [random.randint(0, len(self.episodes) - 1) for _ in range(batch_size)]
        return self._build_batch(indices, max_len)

    def sample_batch_weighted(
        self, batch_size: int, max_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample episodes with probability proportional to episode return.
        Higher-return episodes are sampled more frequently.
        """
        returns = torch.tensor(self.episode_returns, dtype=torch.float32)
        # Shift so all weights are positive
        weights = returns - returns.min() + 1e-6
        indices = torch.multinomial(weights, batch_size, replacement=True).tolist()
        return self._build_batch(indices, max_len)

    def _build_batch(
        self, indices: List[int], max_len: int
    ) -> Dict[str, torch.Tensor]:
        state_dim = self.episodes[0][0].state.shape[0]
        action_dim = self.episodes[0][0].action.shape[0]

        batch_states = torch.zeros(len(indices), max_len, state_dim)
        batch_actions = torch.zeros(len(indices), max_len, action_dim)
        batch_rewards = torch.zeros(len(indices), max_len)
        batch_rtgs = torch.zeros(len(indices), max_len, 1)
        batch_timesteps = torch.zeros(len(indices), max_len, dtype=torch.long)
        batch_mask = torch.zeros(len(indices), max_len, dtype=torch.bool)

        for i, ep_idx in enumerate(indices):
            episode = self.episodes[ep_idx]
            ep_len = len(episode)

            # If episode is longer than max_len, take a random contiguous crop
            if ep_len > max_len:
                start = random.randint(0, ep_len - max_len)
                episode = episode[start : start + max_len]
                ep_len = max_len

            for t, trans in enumerate(episode):
                batch_states[i, t] = trans.state
                batch_actions[i, t] = trans.action
                batch_rewards[i, t] = trans.reward
                batch_rtgs[i, t, 0] = trans.rtg
                batch_timesteps[i, t] = trans.timestep
                batch_mask[i, t] = True

        return {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "rtgs": batch_rtgs,
            "timesteps": batch_timesteps,
            "attention_mask": batch_mask,
        }

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_best_return(self) -> float:
        if not self.episode_returns:
            return 0.0
        return max(self.episode_returns)

    def get_mean_return(self) -> float:
        if not self.episode_returns:
            return 0.0
        return sum(self.episode_returns) / len(self.episode_returns)

    def __len__(self) -> int:
        return len(self.episodes)
