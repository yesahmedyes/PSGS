import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Complete state encoder that combines Gaussian encoding with context features.

    Architecture:
        1. Encode Gaussians using GaussianStateEncoder
        2. Encode context features (iteration, SSIM loss, L1 loss)
        3. Concatenate Gaussian state with context
    """

    def __init__(
        self,
        num_inducing_vectors: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        sh_degree: int = 3,
    ):
        """
        Args:
            num_inducing_vectors: Number of learnable inducing vectors
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            sh_degree: Spherical harmonics degree
        """
        super().__init__()

        # Gaussian encoder
        self.gaussian_encoder = GaussianStateEncoder(
            num_inducing_vectors=num_inducing_vectors,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            sh_degree=sh_degree,
        )

        # Context feature dimension: iteration (1) + ssim_loss (1) + l1_loss (1)
        self.context_dim = 3

        # Final output dimension: Gaussian encoding + context
        self.output_dim = self.gaussian_encoder.get_output_dim() + self.context_dim

    def _encode_context(
        self,
        iteration: int,
        max_iterations: int,
        avg_ssim_loss: float,
        avg_l1_loss: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode context features into a tensor.

        Args:
            iteration: Current training iteration
            max_iterations: Maximum iterations for normalization
            avg_ssim_loss: Running average of recent SSIM losses
            avg_l1_loss: Running average of recent L1 losses
            device: Device to place the tensor on

        Returns:
            Context tensor of shape [context_dim]
        """
        context_features = []

        # Normalized iteration
        iter_normalized = min(iteration / max_iterations, 1.0)
        context_features.append(iter_normalized)

        # Average SSIM loss
        context_features.append(avg_ssim_loss)

        # Average L1 loss
        context_features.append(avg_l1_loss)

        context_tensor = torch.tensor(
            context_features, dtype=torch.float32, device=device
        )

        return context_tensor

    def forward(
        self,
        gaussians,
        iteration: int,
        max_iterations: int,
        avg_ssim_loss: float = 0.0,
        avg_l1_loss: float = 0.0,
    ) -> torch.Tensor:
        """
        Encode complete state including Gaussians and context.

        Args:
            gaussians: GaussianModel instance
            iteration: Current training iteration
            max_iterations: Maximum iterations for normalization
            avg_ssim_loss: Running average of recent SSIM losses
            avg_l1_loss: Running average of recent L1 losses

        Returns:
            State tensor of shape [output_dim]
        """
        # Encode Gaussians
        gaussian_state = self.gaussian_encoder(gaussians)  # [3 * d_model]

        # Encode context
        context_state = self._encode_context(
            iteration=iteration,
            max_iterations=max_iterations,
            avg_ssim_loss=avg_ssim_loss,
            avg_l1_loss=avg_l1_loss,
            device=gaussian_state.device,
        )  # [context_dim]

        # Concatenate Gaussian state with context
        full_state = torch.cat([gaussian_state, context_state], dim=0)  # [output_dim]

        return full_state

    def get_output_dim(self) -> int:
        return self.output_dim


class GaussianStateEncoder(nn.Module):
    """
    Encodes variable-sized sets of 3D Gaussians into fixed-size state vectors
    using cross-attention with learnable inducing vectors.

    Architecture:
        1. Embed Gaussian parameters to d_model
        2. Cross-attend inducing vectors (queries) to Gaussians (keys/values)
        3. Apply max-pool, min-pool, and mean-pool across inducing vectors
        4. Concatenate pooled features
    """

    def __init__(
        self,
        num_inducing_vectors: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        sh_degree: int = 3,
    ):
        """
        Args:
            num_inducing_vectors: Number of learnable inducing vectors
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            sh_degree: Spherical harmonics degree
        """
        super().__init__()

        self.num_inducing_vectors = num_inducing_vectors
        self.d_model = d_model
        self.num_heads = num_heads
        self.sh_degree = sh_degree

        # Compute Gaussian feature dimension
        # xyz (3) + opacity (1) + scaling (3) + rotation (4) + features_dc (3) + features_rest (3 * sh_degree^2 - 3)
        self.gaussian_input_dim = 3 + 1 + 3 + 4 + 3

        if sh_degree > 0:
            num_rest_coeffs = 3 * (sh_degree + 1) ** 2 - 3
            self.gaussian_input_dim += num_rest_coeffs

        # Gaussian embedding layer
        self.gaussian_embed = nn.Linear(self.gaussian_input_dim, d_model)

        # Learnable inducing vectors (queries)
        self.inducing_vectors = nn.Parameter(torch.randn(num_inducing_vectors, d_model))
        nn.init.xavier_uniform_(self.inducing_vectors)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Output dimension: max-pool (d_model) + min-pool (d_model) + mean-pool (d_model)
        self.output_dim = 3 * d_model

    def _extract_gaussian_features(self, gaussians) -> torch.Tensor:
        """
        Extract features from Gaussian model.

        Args:
            gaussians: GaussianModel instance

        Returns:
            Tensor of shape [N, gaussian_input_dim] where N is number of Gaussians
        """
        features = []

        # Position (xyz)
        features.append(gaussians.get_xyz)  # [N, 3]

        # Opacity
        features.append(gaussians.get_opacity)  # [N, 1]

        # Scaling
        features.append(gaussians.get_scaling)  # [N, 3]

        # Rotation
        features.append(gaussians.get_rotation)  # [N, 4]

        # Features DC (first SH coefficient)
        features.append(gaussians.get_features[:, 0, :])  # [N, 3]

        # Features rest (remaining SH coefficients) if degree > 0
        if self.sh_degree > 0 and gaussians.get_features.shape[1] > 1:
            rest_features = gaussians.get_features[:, 1:, :].reshape(
                gaussians.get_features.shape[0], -1
            )
            features.append(rest_features)

        # Concatenate all features
        gaussian_features = torch.cat(features, dim=-1)  # [N, gaussian_input_dim]

        return gaussian_features

    def forward(self, gaussians) -> torch.Tensor:
        """
        Encode Gaussian state to fixed-size vector.

        Args:
            gaussians: GaussianModel instance

        Returns:
            State tensor of shape [output_dim] = [3 * d_model]
        """
        # Extract Gaussian features
        gaussian_features = self._extract_gaussian_features(gaussians)  # [N, input_dim]

        # Embed Gaussians
        gaussian_embedded = self.gaussian_embed(gaussian_features)  # [N, d_model]

        # Prepare inducing vectors as queries
        queries = self.inducing_vectors.unsqueeze(0)  # [1, K, d_model]
        keys_values = gaussian_embedded.unsqueeze(0)  # [1, N, d_model]

        # Cross-attention: inducing vectors attend to Gaussians
        attn_output, _ = self.cross_attention(
            query=queries,
            key=keys_values,
            value=keys_values,
        )  # [1, K, d_model]

        # Layer norm
        attn_output = self.layer_norm(attn_output)  # [1, K, d_model]
        attn_output = attn_output.squeeze(0)  # [K, d_model]

        # Apply three types of pooling across inducing vectors
        max_pooled, _ = torch.max(attn_output, dim=0)  # [d_model]
        min_pooled, _ = torch.min(attn_output, dim=0)  # [d_model]
        mean_pooled = torch.mean(attn_output, dim=0)  # [d_model]

        # Concatenate all pooled features
        gaussian_state = torch.cat(
            [max_pooled, min_pooled, mean_pooled], dim=0
        )  # [3 * d_model]

        return gaussian_state

    def get_output_dim(self) -> int:
        return self.output_dim
