"""
Online Decision Transformer for PGGS learning rate scheduling.

Architecture:
    Interleaved (RTG, state, action) token sequence → causal transformer
    → stochastic action head (Gaussian with squashing to [0.1, 10.0]).

Reuses CausalTransformerBlock, SinusoidalPositionEncoding, and squashing
utilities from ppo_policy.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from pggs.ppo_policy import (
    CausalTransformerBlock,
    SinusoidalPositionEncoding,
    _squash_to_lr_range,
    _squash_log_prob,
    _LOG_MAX_SCALE,
)


class OnlineDecisionTransformer(nn.Module):
    """
    Online Decision Transformer with stochastic action head.

    Token sequence layout (T timesteps → 3T tokens):
        [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ..., RTG_T, s_T, a_T]

    Action prediction for step t comes from the output at position 3*(t-1)+1
    (the s_t token), which causally can only see tokens up to and including s_t
    but NOT a_t.

    Actions are parameterised as Gaussian in unconstrained space, then squashed
    to [1/10, 10] via  a = exp(log(10) * tanh(x)).
    """

    def __init__(
        self,
        state_dim: int = 771,
        action_dim: int = 6,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_episode_len: int = 128,
        device: str = "cuda",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_episode_len = max_episode_len
        self.device = device

        # ── Modality embeddings ──────────────────────────────────────────────
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)

        # Learned timestep embedding (shared across the RTG/state/action triplet)
        self.embed_timestep = nn.Embedding(max_episode_len, hidden_dim)

        # ── Transformer ──────────────────────────────────────────────────────
        self.pos_enc = SinusoidalPositionEncoding(
            hidden_dim, max_len=3 * max_episode_len
        )
        self.embed_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(hidden_dim, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ── Stochastic action head ───────────────────────────────────────────
        self.action_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Rolling token buffer for step-by-step rollout ────────────────────
        self.register_buffer(
            "_token_buffer",
            torch.zeros(3 * max_episode_len, hidden_dim),
        )
        self._buffer_len = 0

        self.to(device)

    # ── Causal mask ──────────────────────────────────────────────────────────

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # ── Full-sequence forward (training) ─────────────────────────────────────

    def forward(
        self,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, nn.Parameter]:
        """
        Full-sequence forward for supervised training on replay buffer.

        Args:
            rtgs:      [batch, T, 1]
            states:    [batch, T, state_dim]
            actions:   [batch, T, action_dim]
            timesteps: [batch, T] (LongTensor, 0-based phase index)
            attention_mask: [batch, T] (BoolTensor, True=valid, False=pad)

        Returns:
            action_means: [batch, T, action_dim]
            action_log_std: [action_dim] (parameter, shared across all positions)
        """
        batch, T, _ = states.shape

        # Embed each modality
        rtg_emb = self.embed_rtg(rtgs)  # [B, T, H]
        state_emb = self.embed_state(states)  # [B, T, H]
        action_emb = self.embed_action(actions)  # [B, T, H]

        # Add timestep embedding
        ts_emb = self.embed_timestep(timesteps)  # [B, T, H]
        rtg_emb = rtg_emb + ts_emb
        state_emb = state_emb + ts_emb
        action_emb = action_emb + ts_emb

        # Interleave: [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ...]
        # Stack → [B, T, 3, H] → [B, 3T, H]
        tokens = (
            torch.stack([rtg_emb, state_emb, action_emb], dim=2)
            .reshape(batch, 3 * T, self.hidden_dim)
        )

        # Positional encoding + LayerNorm
        tokens = self.pos_enc(tokens)
        tokens = self.embed_ln(tokens)

        # Build causal mask
        causal = self._causal_mask(3 * T, tokens.device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # Expand [B, T] → [B, 3T]: each timestep's mask applies to its 3 tokens
            pad_mask_3t = (
                attention_mask.unsqueeze(-1)
                .expand(-1, -1, 3)
                .reshape(batch, 3 * T)
            )  # [B, 3T], True=valid
            # Convert to additive: 0 for valid, -inf for padding
            # Shape [B, 1, 1, 3T] for broadcasting with [3T, 3T] causal mask
            pad_additive = torch.where(
                pad_mask_3t.unsqueeze(1).unsqueeze(2),
                torch.tensor(0.0, device=tokens.device),
                torch.tensor(float("-inf"), device=tokens.device),
            )  # [B, 1, 1, 3T]
            # CausalTransformerBlock uses nn.MultiheadAttention with attn_mask
            # which expects [seq, seq] or [B*nheads, seq, seq].
            # We handle padding by zeroing out padded token embeddings instead,
            # since the causal mask alone is sufficient for correctness.
            tokens = tokens * pad_mask_3t.unsqueeze(-1).float()

        # Run through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, causal)
        tokens = self.final_norm(tokens)

        # Extract outputs at state positions: indices 1, 4, 7, ... (0-indexed)
        state_outputs = tokens[:, 1::3, :]  # [B, T, H]

        # Action distribution parameters
        action_means = self.action_mean_head(state_outputs)  # [B, T, action_dim]

        return action_means, self.action_log_std

    # ── Step-by-step inference (rollout) ─────────────────────────────────────

    def get_action(
        self,
        rtg: float,
        state: torch.Tensor,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action during rollout collection.

        Appends (RTG, state) tokens to the rolling buffer, runs the transformer,
        and samples from the stochastic head. Call ``append_action_to_buffer``
        after applying the action to complete the triplet.

        Args:
            rtg: scalar return-to-go
            state: [state_dim]
            timestep: 0-based phase index

        Returns:
            action:   [action_dim] squashed to [0.1, 10.0]
            log_prob: scalar
        """
        with torch.no_grad():
            ts = torch.tensor([timestep], dtype=torch.long, device=self.device)
            ts_emb = self.embed_timestep(ts).squeeze(0)  # [H]

            rtg_t = torch.tensor([[rtg]], dtype=torch.float32, device=self.device)
            rtg_token = self.embed_rtg(rtg_t).squeeze(0).squeeze(0) + ts_emb  # [H]

            state_token = self.embed_state(state.unsqueeze(0)).squeeze(0) + ts_emb  # [H]

            # Append RTG and state tokens to buffer
            self._append_token(rtg_token)
            self._append_token(state_token)

            # Run buffer through transformer
            seq = self._token_buffer[: self._buffer_len].unsqueeze(0)  # [1, L, H]
            seq = self.pos_enc(seq)
            seq = self.embed_ln(seq)
            mask = self._causal_mask(self._buffer_len, self.device)
            for block in self.blocks:
                seq = block(seq, mask)
            seq = self.final_norm(seq)

            # Last token output (the state token)
            feature = seq[0, -1, :]  # [H]

            # Sample action
            mean = self.action_mean_head(feature)  # [action_dim]
            std = torch.exp(self.action_log_std.clamp(-5, 2))
            dist = Normal(mean, std)
            raw = dist.rsample()  # [action_dim]
            log_prob_raw = dist.log_prob(raw).sum(-1)  # scalar

            action = _squash_to_lr_range(raw)
            log_prob = _squash_log_prob(raw, log_prob_raw)

        return action, log_prob

    def append_action_to_buffer(self, action: torch.Tensor) -> None:
        """
        Embed the chosen action and append to the rolling buffer.
        Call after ``get_action`` and applying the action to the environment.

        Args:
            action: [action_dim] squashed action
        """
        with torch.no_grad():
            # We don't add timestep embedding to the action here because
            # the timestep for this action's triplet was already used for
            # the RTG and state tokens. However, for consistency, we retrieve
            # the timestep from the current buffer position.
            # The triplet index is (buffer_len - 1) // 3 (since we've already
            # appended RTG and state for this step).
            triplet_idx = (self._buffer_len) // 3  # current triplet (0-based)
            # Clamp in case we're near max
            triplet_idx = min(triplet_idx, self.max_episode_len - 1)
            ts = torch.tensor([triplet_idx], dtype=torch.long, device=self.device)
            ts_emb = self.embed_timestep(ts).squeeze(0)

            action_token = (
                self.embed_action(action.unsqueeze(0)).squeeze(0) + ts_emb
            )  # [H]
            self._append_token(action_token)

    def get_action_deterministic(
        self,
        rtg: float,
        state: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        Return the mean action (no sampling). For inference in train.py.

        Args:
            rtg: scalar return-to-go
            state: [state_dim]
            timestep: 0-based phase index

        Returns:
            action: [action_dim] squashed to [0.1, 10.0]
        """
        with torch.no_grad():
            ts = torch.tensor([timestep], dtype=torch.long, device=self.device)
            ts_emb = self.embed_timestep(ts).squeeze(0)

            rtg_t = torch.tensor([[rtg]], dtype=torch.float32, device=self.device)
            rtg_token = self.embed_rtg(rtg_t).squeeze(0).squeeze(0) + ts_emb
            state_token = self.embed_state(state.unsqueeze(0)).squeeze(0) + ts_emb

            self._append_token(rtg_token)
            self._append_token(state_token)

            seq = self._token_buffer[: self._buffer_len].unsqueeze(0)
            seq = self.pos_enc(seq)
            seq = self.embed_ln(seq)
            mask = self._causal_mask(self._buffer_len, self.device)
            for block in self.blocks:
                seq = block(seq, mask)
            seq = self.final_norm(seq)

            feature = seq[0, -1, :]
            mean = self.action_mean_head(feature)
            action = _squash_to_lr_range(mean)

        return action

    # ── Buffer helpers ────────────────────────────────────────────────────────

    def _append_token(self, token: torch.Tensor) -> None:
        """Append a token to the rolling buffer, shifting left if full."""
        max_len = 3 * self.max_episode_len
        if self._buffer_len < max_len:
            self._token_buffer[self._buffer_len] = token
            self._buffer_len += 1
        else:
            self._token_buffer = torch.roll(self._token_buffer, -1, dims=0)
            self._token_buffer[-1] = token

    def reset_episode(self) -> None:
        """Reset the rolling token buffer at the start of a new episode."""
        self._token_buffer.zero_()
        self._buffer_len = 0
