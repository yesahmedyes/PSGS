import torch
import torch.nn as nn
from torchrl.modules import DecisionTransformer


class PolicyNetwork(nn.Module):
    """
    Policy network based on Online Decision Transformer for learning rate scheduling.

    Takes state from StateEncoder and outputs 6 learning rate scaling factors.

    Architecture:
        StateEncoder output -> DecisionTransformer -> Action head -> LR scaling factors
    """

    def __init__(
        self,
        state_dim: int = 771,  # Default: 3 * 256 + 3 from StateEncoder
        num_lr_params: int = 6,  # 6 learning rate scaling factors (xyz, f_dc, f_rest, opacity, scaling, rotation)
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        n_inner: int = 1024,
        activation: str = "gelu",
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        sequence_length: int = 10,  # Number of recent timesteps to keep in context
        device: str = "cuda",
    ):
        """
        Args:
            state_dim: Dimension of state from StateEncoder
            num_lr_params: Number of learning rate scaling factors (6)
            n_embd: Transformer embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_inner: FFN inner dimension
            activation: Activation function
            n_positions: Maximum sequence length
            resid_pdrop: Residual dropout probability
            attn_pdrop: Attention dropout probability
            sequence_length: Number of recent states to maintain in sliding window
            device: Device to place the model on
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_lr_params = num_lr_params
        self.sequence_length = sequence_length
        self.device = device

        # Configure DecisionTransformer
        config = DecisionTransformer.DTConfig(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation=activation,
            n_positions=n_positions,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        # Decision Transformer backbone
        self.decision_transformer = DecisionTransformer(
            state_dim=state_dim,
            action_dim=num_lr_params,
            config=config,
            device=device,
        )

        # Action head: project transformer output to learning rate scaling factors
        self.action_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_embd // 2, num_lr_params),
        ).to(device)

        # Buffers for maintaining sliding window of recent states, actions, returns
        self.register_buffer(
            "state_buffer", torch.zeros(sequence_length, state_dim, device=device)
        )
        self.register_buffer(
            "action_buffer", torch.zeros(sequence_length, num_lr_params, device=device)
        )
        self.register_buffer(
            "return_buffer", torch.zeros(sequence_length, 1, device=device)
        )
        self.buffer_idx = 0
        self.buffer_filled = False

        # Learning rate scaling range: [0.1, 10.0]
        self.lr_scale_min = 0.1
        self.lr_scale_max = 10.0

    def _add_to_buffer(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,
        return_to_go: float = 0.0,
    ):
        """
        Add a new state, action, and return to the sliding window buffer.

        Args:
            state: State tensor of shape [state_dim]
            action: Action tensor of shape [num_lr_params] (can be None for first call)
            return_to_go: Return-to-go scalar value
        """
        # Roll buffers to make room for new entry
        self.state_buffer = torch.roll(self.state_buffer, shifts=-1, dims=0)
        self.action_buffer = torch.roll(self.action_buffer, shifts=-1, dims=0)
        self.return_buffer = torch.roll(self.return_buffer, shifts=-1, dims=0)

        # Add new entries at the end
        self.state_buffer[-1] = state

        if action is not None:
            self.action_buffer[-1] = action

        self.return_buffer[-1] = return_to_go

        # Track if buffer is full
        if self.buffer_idx < self.sequence_length - 1:
            self.buffer_idx += 1
        else:
            self.buffer_filled = True

    def _get_active_sequence(self):
        """
        Get the active portion of the buffer (handles partially filled buffer).

        Returns:
            states: [seq_len, state_dim]
            actions: [seq_len, num_lr_params]
            returns: [seq_len, 1]
            seq_len: Active sequence length
        """
        if self.buffer_filled:
            return (
                self.state_buffer,
                self.action_buffer,
                self.return_buffer,
                self.sequence_length,
            )
        else:
            active_len = self.buffer_idx + 1

            return (
                self.state_buffer[:active_len],
                self.action_buffer[:active_len],
                self.return_buffer[:active_len],
                active_len,
            )

    def forward(
        self,
        state: torch.Tensor,
        return_to_go: float = 0.0,
        use_sequence: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            state: Current state from StateEncoder, shape [state_dim]
            return_to_go: Expected cumulative return (e.g., negative cumulative loss)
            use_sequence: Whether to use sequential context from buffer

        Returns:
            Learning rate scaling factors of shape [num_lr_params] in range [0.1, 10.0]
        """
        # Add current state to buffer (with dummy action for now)
        if use_sequence:
            self._add_to_buffer(state, action=None, return_to_go=return_to_go)

            # Get active sequence from buffer
            states, actions, returns, seq_len = self._get_active_sequence()

            # Add batch dimension: [1, seq_len, dim]
            states = states.unsqueeze(0)  # [1, seq_len, state_dim]
            actions = actions.unsqueeze(0)  # [1, seq_len, num_lr_params]
            returns = returns.unsqueeze(0)  # [1, seq_len, 1]
        else:
            # Single-step mode: just use current state
            states = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            actions = torch.zeros(1, 1, self.num_lr_params, device=self.device)
            returns = torch.tensor([[[return_to_go]]], device=self.device)

        # Pass through Decision Transformer
        # Output shape: [1, seq_len, n_embd]
        dt_output = self.decision_transformer(states, actions, returns)

        # Take the last timestep's output (most recent)
        last_output = dt_output[0, -1, :]  # [n_embd]

        # Project to action space
        raw_actions = self.action_head(last_output)  # [num_lr_params]

        # Apply sigmoid to get range [0, 1], then scale to [0.1, 10.0]
        # sigmoid(x) * (max - min) + min
        lr_scaling_factors = (
            torch.sigmoid(raw_actions) * (self.lr_scale_max - self.lr_scale_min)
            + self.lr_scale_min
        )

        # Update action buffer with the predicted action
        if use_sequence:
            self.action_buffer[-1] = lr_scaling_factors.detach()

        return lr_scaling_factors

    def reset_buffer(self):
        self.state_buffer.zero_()
        self.action_buffer.zero_()
        self.return_buffer.zero_()
        self.buffer_idx = 0
        self.buffer_filled = False

    def get_lr_param_names(self):
        return ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
