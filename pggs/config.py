from dataclasses import dataclass


@dataclass
class PGGSConfig:
    use_pggs: bool = True

    phase_length: int = 12
    update_frequency: int = 1  # How many phases between policy updates.

    start_iteration: int = 0

    num_inducing_vectors: int = 128  # Number of learnable inducing vectors
    state_d_model: int = 256  # Model dimension for state encoder.
    state_num_heads: int = 8  # Number of attention heads in state encoder.
    state_dropout: float = 0.1  # Dropout rate for state encoder.

    train_state_encoder: bool = True  # Whether to train the state encoder
    train_policy_network: bool = False  # Whether to train the policy network.

    num_lr_params: int = 6

    policy_n_embd: int = 256  # Transformer embedding dimension for policy network.
    policy_n_layer: int = 4  # Number of transformer layers in policy network.
    policy_n_head: int = 4  # Number of attention heads in policy network.
    policy_n_inner: int = 1024  # FFN inner dimension in policy network.
    policy_activation: str = "gelu"  # Activation function for policy network.
    policy_n_positions: int = 1024  # Maximum sequence length for policy network.
    policy_resid_pdrop: float = 0.1  # Residual dropout probability for policy network.
    policy_attn_pdrop: float = 0.1  # Attention dropout probability for policy network.
    policy_sequence_length: int = 10  # context window.

    policy_lr: float = 1e-4
    state_encoder_lr: float = 1e-6
    reward_prediction_lr: float = 1e-6  # Learning rate for reward prediction head

    koniq_model_path: str = "koniqplusplus/pretrained_model"
    policy_weight_decay: float = 1e-5
    policy_gradient_clip: float = 1.0

    discount_factor: float = 0.99

    use_reward_normalization: bool = True

    save_policy_checkpoints: bool = True

    state_encoder_checkpoint: str = "checkpoints/state_encoder.pth"
    policy_network_checkpoint: str = "checkpoints/policy_network.pth"

    @property
    def group_mapping(self) -> dict:

        return {
            "xyz": 0,
            "f_dc": 1,
            "f_rest": 2,
            "opacity": 3,
            "scaling": 4,
            "rotation": 5,
        }


default_config = PGGSConfig()
