import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class CausalTransformerBackbone(nn.Module):
    """
    Simple causal transformer. Maintains a rolling token buffer so it can be
    used step-by-step during rollout collection, and also in full-sequence mode
    during the PPO update pass.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.pos_enc = SinusoidalPositionEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Returns an additive mask: -inf above diagonal, 0 on/below
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        x = self.pos_enc(x)
        mask = self._causal_mask(x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# PPO Actor-Critic
# ──────────────────────────────────────────────────────────────────────────────

LR_SCALE_MAX = 10.0
# Range is symmetric in log-space: [1/LR_SCALE_MAX, LR_SCALE_MAX]
_LOG_MAX_SCALE = math.log(LR_SCALE_MAX)


def _squash_to_lr_range(raw: torch.Tensor) -> torch.Tensor:
    """
    Log-space tanh squash from unconstrained space to [1/LR_SCALE_MAX, LR_SCALE_MAX].

    a = exp(log(max_scale) * tanh(x))

    Key values:
        raw → -inf  ->  scale = 1/10  (strong decrease)
        raw =  0    ->  scale = 1     (no change, neutral)
        raw → +inf  ->  scale = 10    (strong increase)
    """
    return torch.exp(_LOG_MAX_SCALE * torch.tanh(raw))


def _squash_log_prob(raw: torch.Tensor, log_prob_raw: torch.Tensor) -> torch.Tensor:
    """
    Adjust log_prob for the log-space tanh transformation.

    For a = exp(c * tanh(x)) where c = log(max_scale):
        da/dx = a * c * (1 - tanh²(x))
        log|da/dx| = c*tanh(x) + log(c) + log(1 - tanh²(x))
    """
    c = _LOG_MAX_SCALE
    log_jac = (
        c * torch.tanh(raw) + math.log(c) + torch.log(1.0 - torch.tanh(raw) ** 2 + 1e-6)
    )
    return log_prob_raw - log_jac.sum(dim=-1)


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic for PGGS learning rate scheduling.

    Supports two recurrent backbones:
        - "gru":         GRU with per-episode hidden state.
        - "transformer": Causal transformer that processes a rolling window of
                         states during inference and the full episode sequence
                         during PPO update.

    Actions are 6 LR scaling factors [xyz, f_dc, f_rest, opacity, scaling,
    rotation] parameterized as Gaussian in unconstrained space, then squashed
    to [1/LR_SCALE_MAX, LR_SCALE_MAX] via log-space tanh:

        a = exp(log(max_scale) * tanh(x))

    This gives a symmetric multiplicative range around 1 (neutral):
        raw=0  ->  scale=1   (no change)
        raw>0  ->  scale>1   (increase LR)
        raw<0  ->  scale<1   (decrease LR)
    """

    def __init__(
        self,
        state_dim: int = 771,
        action_dim: int = 6,
        backbone: str = "gru",
        hidden_dim: int = 256,
        # GRU-specific
        gru_num_layers: int = 2,
        gru_dropout: float = 0.0,
        # Transformer-specific
        transformer_n_heads: int = 4,
        transformer_n_layers: int = 4,
        transformer_ffn_dim: int = 512,
        transformer_dropout: float = 0.1,
        transformer_max_seq_len: int = 64,
        device: str = "cuda",
    ):
        super().__init__()

        assert backbone in ("gru", "transformer"), (
            f"backbone must be 'gru' or 'transformer', got '{backbone}'"
        )

        self.backbone_type = backbone
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.transformer_max_seq_len = transformer_max_seq_len

        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # Backbone
        if backbone == "gru":
            self.gru_num_layers = gru_num_layers
            self.backbone = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            )
        else:
            self.backbone = CausalTransformerBackbone(
                d_model=hidden_dim,
                n_heads=transformer_n_heads,
                n_layers=transformer_n_layers,
                ffn_dim=transformer_ffn_dim,
                dropout=transformer_dropout,
                max_seq_len=transformer_max_seq_len,
            )
            # Rolling token buffer for step-by-step inference
            self.register_buffer(
                "_token_buffer",
                torch.zeros(transformer_max_seq_len, hidden_dim),
            )
            self._buffer_len = 0

        # Actor: mean and learned log_std
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.to(device)

    # ── Hidden state helpers ──────────────────────────────────────────────────

    def init_hidden(self) -> Optional[torch.Tensor]:
        """
        Returns zero hidden state for GRU; None for Transformer.
        For GRU shape: [num_layers, 1, hidden_dim]
        """
        if self.backbone_type == "gru":
            return torch.zeros(
                self.gru_num_layers, 1, self.hidden_dim, device=self.device
            )
        else:
            self._reset_token_buffer()
            return None

    def _reset_token_buffer(self):
        """Reset rolling token buffer for Transformer step-by-step inference."""
        if self.backbone_type == "transformer":
            self._token_buffer.zero_()
            self._buffer_len = 0

    def _get_lr_param_names(self):
        return ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _run_backbone_step(
        self, projected: torch.Tensor, hidden: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single-step backbone forward for rollout collection.

        Args:
            projected: [hidden_dim] or [1, 1, hidden_dim]
            hidden: GRU hidden state [n_layers, 1, hidden_dim] or None

        Returns:
            feature: [hidden_dim]
            next_hidden: updated hidden state or None
        """
        if self.backbone_type == "gru":
            x = projected.view(1, 1, self.hidden_dim)  # [1, 1, H]
            out, next_hidden = self.backbone(x, hidden)  # [1, 1, H]
            return out.squeeze(0).squeeze(0), next_hidden
        else:
            # Append to rolling token buffer
            token = projected.view(self.hidden_dim)
            if self._buffer_len < self.transformer_max_seq_len:
                self._token_buffer[self._buffer_len] = token
                self._buffer_len += 1
            else:
                # Roll left and add new token at end
                self._token_buffer = torch.roll(self._token_buffer, -1, dims=0)
                self._token_buffer[-1] = token

            seq = self._token_buffer[: self._buffer_len].unsqueeze(0)  # [1, T, H]
            out = self.backbone(seq)  # [1, T, H]
            return out[0, -1, :], None  # last token's output

    # ── Public API ────────────────────────────────────────────────────────────

    def get_action_and_value(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample an action during rollout collection.

        Args:
            state: [state_dim]
            hidden: GRU hidden [n_layers, 1, H] or None for Transformer

        Returns:
            action:    [action_dim]  in [1/LR_SCALE_MAX, LR_SCALE_MAX]
            log_prob:  scalar
            value:     scalar
            next_hidden
        """
        projected = self.input_proj(state)  # [H]
        feature, next_hidden = self._run_backbone_step(projected, hidden)

        mean = self.actor_mean(feature)  # [action_dim]
        std = torch.exp(self.actor_log_std.clamp(-5, 2))
        dist = Normal(mean, std)
        raw = dist.rsample()  # [action_dim]
        log_prob_raw = dist.log_prob(raw).sum(-1)  # scalar

        action = _squash_to_lr_range(raw)
        log_prob = _squash_log_prob(raw, log_prob_raw)

        value = self.critic(feature).squeeze(-1)  # scalar

        return action, log_prob, value, next_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions_squashed: torch.Tensor,
        init_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log_probs, values, and entropy for a full rollout sequence.
        Used during PPO update.

        Args:
            states:           [seq_len, state_dim]  (one full episode or minibatch)
            actions_squashed: [seq_len, action_dim] in [1/LR_SCALE_MAX, LR_SCALE_MAX]
            init_hidden:      Initial GRU hidden state [n_layers, 1, H] or None

        Returns:
            log_probs:  [seq_len]
            values:     [seq_len]
            entropy:    scalar
        """
        projected = self.input_proj(states)  # [T, H]

        if self.backbone_type == "gru":
            x = projected.unsqueeze(0)  # [1, T, H]
            features, _ = self.backbone(x, init_hidden)  # [1, T, H]
            features = features.squeeze(0)  # [T, H]
        else:
            x = projected.unsqueeze(0)  # [1, T, H]
            features = self.backbone(x).squeeze(0)  # [T, H]

        means = self.actor_mean(features)  # [T, action_dim]
        std = torch.exp(self.actor_log_std.clamp(-5, 2))
        dist = Normal(means, std)

        # Invert the log-space tanh squash to recover raw actions:
        #   a = exp(c * tanh(x))  =>  tanh(x) = log(a) / c  =>  x = atanh(log(a) / c)
        c = _LOG_MAX_SCALE
        log_a = torch.log(actions_squashed.clamp(min=1e-6))
        raw = torch.atanh((log_a / c).clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        log_prob_raw = dist.log_prob(raw).sum(-1)  # [T]
        log_probs = _squash_log_prob(raw, log_prob_raw)  # [T]

        values = self.critic(features).squeeze(-1)  # [T]
        entropy = dist.entropy().sum(-1).mean()  # scalar

        return log_probs, values, entropy

    def get_action_deterministic(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns the mean action (no sampling). Used in train.py for inference.

        Args:
            state: [state_dim]
            hidden: GRU hidden or None

        Returns:
            action:     [action_dim] in [1/LR_SCALE_MAX, LR_SCALE_MAX]
            next_hidden
        """
        with torch.no_grad():
            projected = self.input_proj(state)
            feature, next_hidden = self._run_backbone_step(projected, hidden)
            mean = self.actor_mean(feature)
            action = _squash_to_lr_range(mean)
        return action, next_hidden

    def reset_episode(self):
        """
        Reset any per-episode state. Call at the start of each new episode.
        For GRU: caller should pass fresh init_hidden().
        For Transformer: resets rolling token buffer.
        """
        if self.backbone_type == "transformer":
            self._reset_token_buffer()
