from dataclasses import dataclass


@dataclass
class PGGSConfig:
    use_pggs: bool = True

    phase_length: int = 10
    update_frequency: int = 1  # How many phases between policy updates.

    start_iteration: int = 0

    # ── State Encoder ─────────────────────────────────────────────────────────
    num_inducing_vectors: int = 128  # Number of learnable inducing vectors
    state_d_model: int = 256  # Model dimension for state encoder.
    state_num_heads: int = 8  # Number of attention heads in state encoder.
    state_dropout: float = 0.1  # Dropout rate for state encoder.

    # Whether to append context scalars (iter, ssim, l1) to Gaussian encoding.
    # True  -> state_dim = 3 * state_d_model + 3  (e.g. 771)
    # False -> state_dim = 3 * state_d_model       (e.g. 768)
    use_context: bool = True

    train_state_encoder: bool = True  # Whether to train the state encoder
    train_policy_network: bool = False  # Whether to train the policy network (DT).

    num_lr_params: int = 6

    # ── Legacy DT Policy (PolicyNetwork / train.py inference) ────────────────
    policy_n_embd: int = 256  # Transformer embedding dimension for policy network.
    policy_n_layer: int = 4  # Number of transformer layers in policy network.
    policy_n_head: int = 4  # Number of attention heads in policy network.
    policy_n_inner: int = 1024  # FFN inner dimension in policy network.
    policy_activation: str = "gelu"  # Activation function for policy network.
    policy_n_positions: int = 1024  # Maximum sequence length for policy network.
    policy_resid_pdrop: float = 0.1  # Residual dropout probability for policy network.
    policy_attn_pdrop: float = 0.1  # Attention dropout probability for policy network.
    policy_sequence_length: int = 10  # context window.

    # ── PPO Actor-Critic ──────────────────────────────────────────────────────
    # Backbone choice: "gru" or "transformer"
    policy_backbone: str = "gru"

    # Shared hidden dimension for both backbones
    ppo_hidden_dim: int = 256

    # GRU-specific
    gru_num_layers: int = 2
    gru_dropout: float = 0.0

    # Simple causal Transformer-specific (not Decision Transformer)
    transformer_n_heads: int = 4
    transformer_n_layers: int = 4
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1
    transformer_seq_len: int = 128  # context window (rolling buffer length)

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    ppo_clip_epsilon: float = 0.2
    ppo_value_coeff: float = 0.5
    ppo_entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 32
    num_ppo_episodes: int = 100

    # ── Learning rates & optimisation ────────────────────────────────────────
    policy_lr: float = 1e-4
    state_encoder_lr: float = 1e-6
    reward_prediction_lr: float = 1e-6  # Learning rate for reward prediction head

    policy_weight_decay: float = 1e-5
    policy_gradient_clip: float = 1.0

    discount_factor: float = 0.99

    use_reward_normalization: bool = True

    # ── Online Decision Transformer (ODT) ─────────────────────────────────────
    # Architecture
    odt_hidden_dim: int = 256
    odt_n_heads: int = 4
    odt_n_layers: int = 4
    odt_ffn_dim: int = 512
    odt_dropout: float = 0.1
    odt_max_episode_len: int = 128  # max phases per episode

    # Replay buffer
    odt_replay_buffer_size: int = 100  # max episodes in buffer
    odt_rtg_discount: float = 1.0  # discount for RTG computation

    # Training
    odt_updates_per_episode: int = 10  # gradient steps per collected episode
    odt_batch_size: int = 8  # episodes per training batch
    odt_context_len: int = 64  # max sequence length for training crops
    odt_lr: float = 1e-4
    odt_weight_decay: float = 1e-4
    odt_gradient_clip: float = 0.25

    # Exploration / RTG targeting
    odt_target_rtg: float = 0.8  # initial target RTG for data collection
    odt_rtg_adapt: bool = True  # adapt target RTG based on observed returns
    odt_rtg_adapt_factor: float = 1.1  # target = best_return * factor

    # Loss
    odt_entropy_coeff: float = 0.01
    odt_use_weighted_sampling: bool = True

    # Checkpoints
    odt_policy_checkpoint: str = "checkpoints/odt_policy.pth"
    odt_replay_buffer_checkpoint: str = "checkpoints/replay_buffer.pth"

    # Warmup
    odt_warmup_episodes: int = 5  # collect before starting ODT updates

    # ── Misc ──────────────────────────────────────────────────────────────────
    koniq_model_path: str = "koniqplusplus/pretrained_model"
    save_policy_checkpoints: bool = True

    state_encoder_checkpoint: str = "checkpoints/state_encoder.pth"
    policy_network_checkpoint: str = "checkpoints/policy_network.pth"
    ppo_policy_checkpoint: str = "checkpoints/ppo_policy.pth"

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
