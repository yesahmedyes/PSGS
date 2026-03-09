#
# PPO training script for PGGS policy learning.
#
# One episode = one full Gaussian training run on one scene.
# The policy receives a 771-D (or 768-D without context) state from the frozen
# (or co-trained) StateEncoder and outputs 6 LR scaling factors per phase.
# Reward = KonIQ++ overall quality score averaged over all training views.
#

import os
import sys
import uuid
from collections import namedtuple
from random import randint
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from torchvision.transforms.functional import resize, normalize
from tqdm import tqdm

from pggs.config import PGGSConfig

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor
from pggs.state_encoder import StateEncoder
from pggs.ppo_policy import PPOActorCritic
from pggs.utils import apply_lr_scaling

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Transition storage
# ──────────────────────────────────────────────────────────────────────────────

Transition = namedtuple(
    "Transition",
    [
        "state",              # [state_dim]  encoded state (used when train_se=False)
        "gaussian_features",  # [N, feat_dim] or None (stored when train_se=True)
        "context",            # [3] or None (iter_norm, ssim, l1; used when train_se=True)
        "iteration",          # int  (for re-encoding; used when train_se=True)
        "max_iterations",     # int
        "action",             # [action_dim]
        "log_prob",           # scalar
        "value",              # scalar
        "reward",             # scalar
        "done",               # bool
        "init_hidden",        # GRU hidden at start of this step, or None
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# KonIQ++ helpers  (mirrored from train_se.py)
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_image_for_koniq(image_tensor: torch.Tensor) -> torch.Tensor:
    """Resize and normalize [3, H, W] image for KonIQ++."""
    image_resized = resize(image_tensor, (480, 640))
    image_normalized = normalize(
        image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return image_normalized


def evaluate_with_koniq(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    koniq_model,
    k,
    b,
    device: str = "cuda",
) -> float:
    """
    Render all training views and return the averaged KonIQ++ overall score
    (normalised to [0, 1]).

    Returns:
        scalar float reward
    """
    train_cameras = scene.getTrainCameras()
    all_scores = []

    with torch.no_grad():
        for viewpoint_cam in train_cameras:
            pose = gaussians.get_RT(viewpoint_cam.uid)
            render_pkg = render_fn(
                viewpoint_cam, gaussians, pipe, background, camera_pose=pose
            )
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            preprocessed = _preprocess_image_for_koniq(rendered_image)
            koniq_output = koniq_model(preprocessed.unsqueeze(0))  # [1, 5]
            overall_score = koniq_output[0, 0].item() * k[0] + b[0]
            all_scores.append(overall_score / 100.0)

    return float(np.mean(all_scores))


# ──────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    dataset,
    opt,
    pipe,
    pggs_config: PGGSConfig,
    state_encoder: StateEncoder,
    policy: PPOActorCritic,
    koniq_model,
    koniq_k,
    koniq_b,
    train_state_encoder: bool = False,
    device: str = "cuda",
) -> Tuple[List[Transition], float]:
    """
    Run one full Gaussian training episode and collect PPO transitions.

    One episode = one full training run on the scene (``opt.iterations``
    iterations).  A transition is recorded at the end of every phase
    (``pggs_config.phase_length`` views).

    Args:
        train_state_encoder: If True, stores raw Gaussian features so the
                             encoder can be fine-tuned during the PPO update.

    Returns:
        rollout:      list of Transition namedtuples
        final_reward: KonIQ++ score at episode end
    """
    # ── Scene & Gaussians initialisation ─────────────────────────────────────
    gaussians = GaussianModel(dataset.sh_degree)

    confidence_path = os.path.join(
        dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy"
    )
    if opt.pp_optimizer and os.path.exists(confidence_path):
        confidence_np = np.load(confidence_path)
        confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
        confidence_norm = torch.sigmoid(confidence_tensor)
        lr_modifiers = (1.0 - confidence_norm) * (100 - 1) + 1
        scene = Scene(dataset, gaussians)
        gaussians.training_setup_pp(opt, lr_modifiers)
    else:
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # ── Policy state ──────────────────────────────────────────────────────────
    policy.train()
    state_encoder.eval()  # encoder in eval unless updated below during PPO

    hidden = policy.init_hidden()
    policy.reset_episode()

    # ── Phase tracking ────────────────────────────────────────────────────────
    rollout: List[Transition] = []
    phase_ssim_losses: List[float] = []
    phase_l1_losses: List[float] = []
    views_in_phase = 0
    phase_counter = 0
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    for iteration in range(1, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        if not opt.optim_pose:
            gaussians.P.requires_grad_(False)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Phase boundary: observe → act → reward ────────────────────────────
        if (
            iteration >= pggs_config.start_iteration
            and views_in_phase >= pggs_config.phase_length
        ):
            avg_ssim = (
                sum(phase_ssim_losses) / len(phase_ssim_losses)
                if phase_ssim_losses
                else 0.0
            )
            avg_l1 = (
                sum(phase_l1_losses) / len(phase_l1_losses)
                if phase_l1_losses
                else 0.0
            )

            # ── Encode state ──────────────────────────────────────────────────
            with torch.no_grad():
                state = state_encoder(
                    gaussians=gaussians,
                    iteration=iteration,
                    max_iterations=opt.iterations,
                    avg_ssim_loss=avg_ssim,
                    avg_l1_loss=avg_l1,
                )  # [state_dim]

            # Store raw Gaussian features if encoder will be fine-tuned
            if train_state_encoder:
                gauss_features = (
                    state_encoder.gaussian_encoder
                    ._extract_gaussian_features(gaussians)
                    .detach()
                    .cpu()
                )
                ctx = torch.tensor(
                    [
                        min(iteration / opt.iterations, 1.0),
                        avg_ssim,
                        avg_l1,
                    ],
                    dtype=torch.float32,
                )
            else:
                gauss_features = None
                ctx = None

            # ── Sample action ─────────────────────────────────────────────────
            stored_hidden = (
                hidden.detach().clone() if hidden is not None else None
            )

            with torch.no_grad():
                action, log_prob, value, hidden = policy.get_action_and_value(
                    state, hidden
                )

            # ── Apply LR scaling ──────────────────────────────────────────────
            apply_lr_scaling(
                optimizer=gaussians.optimizer,
                action=action,
                group_mapping=pggs_config.group_mapping,
            )

            # ── Reward ────────────────────────────────────────────────────────
            done = (iteration == opt.iterations)
            reward = evaluate_with_koniq(
                gaussians=gaussians,
                scene=scene,
                render_fn=render,
                pipe=pipe,
                background=background,
                koniq_model=koniq_model,
                k=koniq_k,
                b=koniq_b,
                device=device,
            )

            rollout.append(
                Transition(
                    state=state.cpu(),
                    gaussian_features=gauss_features,
                    context=ctx,
                    iteration=iteration,
                    max_iterations=opt.iterations,
                    action=action.detach().cpu(),
                    log_prob=log_prob.detach().cpu(),
                    value=value.detach().cpu(),
                    reward=reward,
                    done=done,
                    init_hidden=stored_hidden.cpu() if stored_hidden is not None else None,
                )
            )

            phase_counter += 1
            views_in_phase = 0
            phase_ssim_losses = []
            phase_l1_losses = []

        # ── Normal Gaussian training step ─────────────────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        bg = (
            torch.rand((3,), device=device)
            if opt.random_background
            else background
        )
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        phase_ssim_losses.append((1.0 - ssim_value).item())
        phase_l1_losses.append(Ll1.item())
        views_in_phase += 1

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    # Final reward after the last iteration
    final_reward = evaluate_with_koniq(
        gaussians=gaussians,
        scene=scene,
        render_fn=render,
        pipe=pipe,
        background=background,
        koniq_model=koniq_model,
        k=koniq_k,
        b=koniq_b,
        device=device,
    )

    return rollout, final_reward


# ──────────────────────────────────────────────────────────────────────────────
# GAE and PPO update
# ──────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rollout: List[Transition],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalised Advantage Estimation and discounted returns.

    Returns:
        advantages: [T]
        returns:    [T]
    """
    T = len(rollout)
    rewards = torch.tensor([t.reward for t in rollout], dtype=torch.float32)
    values = torch.stack([t.value for t in rollout])
    dones = torch.tensor([float(t.done) for t in rollout], dtype=torch.float32)

    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1].item()
        next_done = dones[t].item()
        delta = rewards[t].item() + gamma * next_value * (1.0 - next_done) - values[t].item()
        last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: PPOActorCritic,
    state_encoder: StateEncoder,
    policy_optimizer: torch.optim.Optimizer,
    state_encoder_optimizer: Optional[torch.optim.Optimizer],
    rollout: List[Transition],
    pggs_config: PGGSConfig,
    train_state_encoder: bool = False,
    device: str = "cuda",
) -> dict:
    """
    Perform PPO update over the collected rollout.

    When train_state_encoder=True, Gaussian features are re-encoded inside the
    update loop so gradients flow into the state encoder.

    Returns:
        dict with scalar loss values for logging.
    """
    T = len(rollout)
    if T == 0:
        return {}

    advantages, returns = compute_gae(
        rollout,
        gamma=pggs_config.discount_factor,
        gae_lambda=pggs_config.gae_lambda,
    )

    if pggs_config.use_reward_normalization and T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Precompute states (re-encode if needed)
    if train_state_encoder:
        state_encoder.train()
    else:
        state_encoder.eval()

    policy.train()

    # Collect tensors once (states will be re-computed inside if train_se=True)
    old_log_probs = torch.stack([t.log_prob for t in rollout]).to(device)
    old_actions = torch.stack([t.action for t in rollout]).to(device)
    advantages_d = advantages.to(device)
    returns_d = returns.to(device)

    # For GRU: we keep the initial hidden of the episode (first transition)
    episode_init_hidden = rollout[0].init_hidden
    if episode_init_hidden is not None:
        episode_init_hidden = episode_init_hidden.to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(pggs_config.ppo_epochs):
        indices = torch.randperm(T)

        for start in range(0, T, pggs_config.ppo_minibatch_size):
            mb_idx = indices[start : start + pggs_config.ppo_minibatch_size]

            # ── Build state batch (potentially re-encoding) ───────────────────
            if train_state_encoder:
                states_list = []
                for idx in mb_idx.tolist():
                    t = rollout[idx]
                    gf = t.gaussian_features.to(device)   # [N, feat_dim]
                    ctx = t.context.to(device)             # [3]
                    # Re-run encoder: embed -> cross-attention -> pool
                    gauss_state = _run_gaussian_encoder(state_encoder, gf)
                    if state_encoder.use_context:
                        s = torch.cat([gauss_state, ctx], dim=0)
                    else:
                        s = gauss_state
                    states_list.append(s)
                mb_states = torch.stack(states_list)      # [mb, state_dim]
            else:
                mb_states = torch.stack(
                    [rollout[i.item()].state for i in mb_idx]
                ).to(device)

            mb_old_log_probs = old_log_probs[mb_idx]
            mb_old_actions = old_actions[mb_idx]
            mb_advantages = advantages_d[mb_idx]
            mb_returns = returns_d[mb_idx]

            # ── Forward pass ──────────────────────────────────────────────────
            # For GRU minibatches we process each sample independently
            # (no hidden state passed) — acceptable since the policy acts on
            # individual steps and the GRU history is already baked into the
            # collected states via the rollout hidden states.
            log_probs, values, entropy = policy.evaluate_actions(
                mb_states, mb_old_actions, init_hidden=None
            )

            # ── PPO losses ────────────────────────────────────────────────────
            ratio = torch.exp(log_probs - mb_old_log_probs.detach())
            surr1 = ratio * mb_advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - pggs_config.ppo_clip_epsilon,
                    1.0 + pggs_config.ppo_clip_epsilon,
                )
                * mb_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, mb_returns)
            total_loss = (
                policy_loss
                + pggs_config.ppo_value_coeff * value_loss
                - pggs_config.ppo_entropy_coeff * entropy
            )

            policy_optimizer.zero_grad()
            if state_encoder_optimizer is not None:
                state_encoder_optimizer.zero_grad()

            total_loss.backward()

            nn.utils.clip_grad_norm_(
                policy.parameters(), pggs_config.policy_gradient_clip
            )
            if train_state_encoder:
                nn.utils.clip_grad_norm_(
                    state_encoder.parameters(), pggs_config.policy_gradient_clip
                )

            policy_optimizer.step()
            if state_encoder_optimizer is not None:
                state_encoder_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    n = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
    }


def _run_gaussian_encoder(
    state_encoder: StateEncoder, gaussian_features: torch.Tensor
) -> torch.Tensor:
    """
    Re-run the Gaussian encoding part of StateEncoder on pre-extracted
    features (used during PPO update when fine-tuning the encoder).

    Args:
        state_encoder: StateEncoder instance
        gaussian_features: [N, gaussian_input_dim]

    Returns:
        gaussian_state: [3 * d_model]
    """
    enc = state_encoder.gaussian_encoder

    # Embed
    embedded = enc.gaussian_embed(gaussian_features)  # [N, d_model]

    # Cross-attention with inducing vectors
    queries = enc.inducing_vectors.unsqueeze(0)  # [1, K, d_model]
    keys_values = embedded.unsqueeze(0)          # [1, N, d_model]
    attn_output, _ = enc.cross_attention(
        query=queries, key=keys_values, value=keys_values
    )  # [1, K, d_model]
    attn_output = enc.layer_norm(attn_output).squeeze(0)  # [K, d_model]

    max_pooled, _ = torch.max(attn_output, dim=0)
    min_pooled, _ = torch.min(attn_output, dim=0)
    mean_pooled = torch.mean(attn_output, dim=0)

    return torch.cat([max_pooled, min_pooled, mean_pooled], dim=0)  # [3 * d_model]


# ──────────────────────────────────────────────────────────────────────────────
# Outer training loop
# ──────────────────────────────────────────────────────────────────────────────

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_ppo(
    dataset,
    opt,
    pipe,
    num_episodes: int,
    scene_list: Optional[List[str]],
    policy_backbone: str,
    use_context: bool,
    train_state_encoder_flag: bool,
    load_state_encoder_path: Optional[str],
    load_ppo_policy_path: Optional[str],
    tb_writer,
    state_encoder_lr_override: Optional[float] = None,
    device: str = "cuda",
):
    """
    Outer PPO training loop.

    Args:
        dataset:               Dataset params (from ModelParams.extract)
        opt:                   Optimisation params
        pipe:                  Pipeline params
        num_episodes:          Number of PPO episodes
        scene_list:            List of additional source_paths; if provided,
                               episodes cycle through these scenes. If None,
                               the single dataset.source_path is used.
        policy_backbone:       "gru" or "transformer"
        use_context:           Whether to include context scalars in state
        train_state_encoder_flag: Whether to fine-tune the state encoder
        load_state_encoder_path:  Checkpoint path or None
        load_ppo_policy_path:     Checkpoint path or None
        tb_writer:             TensorBoard SummaryWriter or None
    """
    pggs_config = PGGSConfig()
    pggs_config.policy_backbone = policy_backbone
    pggs_config.use_context = use_context

    if state_encoder_lr_override is not None:
        pggs_config.state_encoder_lr = state_encoder_lr_override

    # ── State Encoder ─────────────────────────────────────────────────────────
    state_encoder = StateEncoder(
        num_inducing_vectors=pggs_config.num_inducing_vectors,
        d_model=pggs_config.state_d_model,
        num_heads=pggs_config.state_num_heads,
        dropout=pggs_config.state_dropout,
        sh_degree=dataset.sh_degree,
        use_context=use_context,
    ).to(device)

    if load_state_encoder_path and os.path.exists(load_state_encoder_path):
        print(f"Loading state encoder from {load_state_encoder_path}")
        ckpt = torch.load(load_state_encoder_path, map_location=device)
        state_key = "state_encoder" if "state_encoder" in ckpt else None
        state_encoder.load_state_dict(ckpt[state_key] if state_key else ckpt)
        print("State encoder loaded.")
    else:
        if load_state_encoder_path:
            print(f"Warning: state encoder checkpoint not found at {load_state_encoder_path}")
        print("Initialising state encoder from scratch.")

    if not train_state_encoder_flag:
        state_encoder.eval()
        for p in state_encoder.parameters():
            p.requires_grad_(False)

    state_dim = state_encoder.get_output_dim()

    # ── PPO Policy ────────────────────────────────────────────────────────────
    policy = PPOActorCritic(
        state_dim=state_dim,
        action_dim=pggs_config.num_lr_params,
        backbone=policy_backbone,
        hidden_dim=pggs_config.ppo_hidden_dim,
        gru_num_layers=pggs_config.gru_num_layers,
        gru_dropout=pggs_config.gru_dropout,
        transformer_n_heads=pggs_config.transformer_n_heads,
        transformer_n_layers=pggs_config.transformer_n_layers,
        transformer_ffn_dim=pggs_config.transformer_ffn_dim,
        transformer_dropout=pggs_config.transformer_dropout,
        transformer_max_seq_len=pggs_config.transformer_seq_len,
        device=device,
    )

    if load_ppo_policy_path and os.path.exists(load_ppo_policy_path):
        print(f"Loading PPO policy from {load_ppo_policy_path}")
        ckpt = torch.load(load_ppo_policy_path, map_location=device)
        policy_key = "policy" if "policy" in ckpt else None
        policy.load_state_dict(ckpt[policy_key] if policy_key else ckpt)
        print("PPO policy loaded.")
    else:
        if load_ppo_policy_path:
            print(f"Warning: PPO policy checkpoint not found at {load_ppo_policy_path}")
        print("Initialising PPO policy from scratch.")

    # ── Optimisers ────────────────────────────────────────────────────────────
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=pggs_config.policy_lr,
        weight_decay=pggs_config.policy_weight_decay,
    )

    state_encoder_optimizer = None
    if train_state_encoder_flag:
        state_encoder_optimizer = torch.optim.Adam(
            state_encoder.parameters(),
            lr=pggs_config.state_encoder_lr,
        )

    # ── KonIQ++ model ─────────────────────────────────────────────────────────
    from koniqplusplus.IQAmodel import Model_Joint

    print(f"Loading KonIQ++ model from {pggs_config.koniq_model_path} ...")
    koniq_model = Model_Joint().to(device)
    koniq_ckpt = torch.load(pggs_config.koniq_model_path, map_location=device)
    koniq_model.load_state_dict(koniq_ckpt["model"])
    koniq_k = koniq_ckpt["k"]
    koniq_b = koniq_ckpt["b"]
    koniq_model.eval()
    print("KonIQ++ loaded.")

    # ── Scene list ────────────────────────────────────────────────────────────
    if not scene_list:
        scene_list = [dataset.source_path]
    n_scenes = len(scene_list)

    # ── CSV loss log ──────────────────────────────────────────────────────────
    csv_path = os.path.join(dataset.model_path, "ppo_losses.csv")
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(csv_path, "w") as _f:
        _f.write("episode,epoch,scene,policy_loss,value_loss,entropy,final_reward\n")

    # ── Episode loop ──────────────────────────────────────────────────────────
    progress_bar = tqdm(range(num_episodes), desc="PPO episodes")

    for episode in range(num_episodes):
        scene_path = scene_list[episode % len(scene_list)]
        # Swap source path for this episode if cycling scenes
        episode_dataset = _clone_dataset_with_scene(dataset, scene_path)

        print(
            f"\n{'='*60}\nEpisode {episode + 1}/{num_episodes}  "
            f"scene: {scene_path}\n{'='*60}"
        )

        rollout, final_reward = run_episode(
            dataset=episode_dataset,
            opt=opt,
            pipe=pipe,
            pggs_config=pggs_config,
            state_encoder=state_encoder,
            policy=policy,
            koniq_model=koniq_model,
            koniq_k=koniq_k,
            koniq_b=koniq_b,
            train_state_encoder=train_state_encoder_flag,
            device=device,
        )

        print(
            f"  Episode {episode + 1}: {len(rollout)} phases, "
            f"final KonIQ++ reward = {final_reward:.4f}"
        )

        losses = {}
        if len(rollout) > 0:
            losses = ppo_update(
                policy=policy,
                state_encoder=state_encoder,
                policy_optimizer=policy_optimizer,
                state_encoder_optimizer=state_encoder_optimizer,
                rollout=rollout,
                pggs_config=pggs_config,
                train_state_encoder=train_state_encoder_flag,
                device=device,
            )
            print(
                f"  PPO update — policy_loss: {losses.get('policy_loss', 0):.4f}  "
                f"value_loss: {losses.get('value_loss', 0):.4f}  "
                f"entropy: {losses.get('entropy', 0):.4f}"
            )

        if tb_writer:
            tb_writer.add_scalar("ppo/final_reward", final_reward, episode)
            tb_writer.add_scalar("ppo/n_phases", len(rollout), episode)
            for k, v in losses.items():
                tb_writer.add_scalar(f"ppo/{k}", v, episode)

        # ── CSV logging ────────────────────────────────────────────────────────
        epoch = episode // n_scenes + 1
        # Derive a short readable label from the scene path entry, e.g.
        # "/data/tanks_templates:output/horse" → "tanks_templates/horse"
        raw_label = scene_path
        if ":" in raw_label:
            parts = raw_label.split(":", 1)
            scene_label = "/".join(os.path.normpath(p).split(os.sep)[-1] for p in parts)
        else:
            components = os.path.normpath(raw_label).split(os.sep)
            scene_label = "/".join(components[-2:]) if len(components) >= 2 else components[-1]
        with open(csv_path, "a") as _f:
            _f.write(
                f"{episode + 1},{epoch},{scene_label},"
                f"{losses.get('policy_loss', float('nan')):.6f},"
                f"{losses.get('value_loss', float('nan')):.6f},"
                f"{losses.get('entropy', float('nan')):.6f},"
                f"{final_reward:.6f}\n"
            )

        progress_bar.update(1)

        # ── Checkpointing ──────────────────────────────────────────────────────
        save_every = max(1, num_episodes // 10)
        if (episode + 1) % save_every == 0 or episode == num_episodes - 1:
            _save_checkpoints(
                policy=policy,
                state_encoder=state_encoder,
                policy_optimizer=policy_optimizer,
                state_encoder_optimizer=state_encoder_optimizer,
                episode=episode,
                pggs_config=pggs_config,
                train_se=train_state_encoder_flag,
            )

    progress_bar.close()
    print("\nPPO training complete.")


def _clone_dataset_with_scene(dataset, entry: str):
    """
    Return a shallow copy of dataset with updated source_path (and optionally
    model_path).

    entry can be either:
        "source_path"               — only source_path is updated
        "source_path:model_path"    — both source_path and model_path updated
                                      (required when init_geo was pre-run per scene)
    """
    import copy
    d = copy.copy(dataset)
    if ":" in entry:
        source, model = entry.split(":", 1)
        d.source_path = os.path.abspath(source)
        d.model_path = os.path.abspath(model)
    else:
        d.source_path = os.path.abspath(entry)
    return d


def _save_checkpoints(
    policy: PPOActorCritic,
    state_encoder: StateEncoder,
    policy_optimizer,
    state_encoder_optimizer,
    episode: int,
    pggs_config: PGGSConfig,
    train_se: bool,
):
    os.makedirs("checkpoints", exist_ok=True)

    policy_ckpt = {
        "policy": policy.state_dict(),
        "optimizer": policy_optimizer.state_dict(),
        "episode": episode,
        "backbone": pggs_config.policy_backbone,
        "hidden_dim": pggs_config.ppo_hidden_dim,
        "action_dim": pggs_config.num_lr_params,
    }
    torch.save(policy_ckpt, pggs_config.ppo_policy_checkpoint)
    print(f"  PPO policy saved → {pggs_config.ppo_policy_checkpoint}")

    if train_se:
        se_ckpt = {
            "state_encoder": state_encoder.state_dict(),
            "optimizer": state_encoder_optimizer.state_dict() if state_encoder_optimizer else None,
            "episode": episode,
            "use_context": pggs_config.use_context,
        }
        torch.save(se_ckpt, pggs_config.state_encoder_checkpoint)
        print(f"  State encoder saved → {pggs_config.state_encoder_checkpoint}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="PPO policy training for PGGS")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # PPO-specific args
    parser.add_argument(
        "--policy_backbone",
        type=str,
        default="gru",
        choices=["gru", "transformer"],
        help="Recurrent backbone for PPO actor-critic: 'gru' or 'transformer'",
    )
    parser.add_argument(
        "--use_context",
        action="store_true",
        default=True,
        help="Include context scalars (iter, ssim, l1) in the state encoding",
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        default=False,
        help="Exclude context scalars — state is Gaussian encoding only",
    )
    parser.add_argument(
        "--train_state_encoder",
        action="store_true",
        default=False,
        help="Fine-tune state encoder during PPO training",
    )
    parser.add_argument(
        "--no_train_state_encoder",
        action="store_true",
        default=False,
        help="Keep state encoder frozen during PPO training (default)",
    )
    parser.add_argument(
        "--load_state_encoder",
        type=str,
        default=None,
        help="Path to pre-trained state encoder checkpoint",
    )
    parser.add_argument(
        "--load_ppo_policy",
        type=str,
        default=None,
        help="Path to pre-trained PPO policy checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of PPO episodes to train",
    )
    parser.add_argument(
        "--scene_list",
        nargs="+",
        type=str,
        default=None,
        help=(
            "List of scene source_path directories to cycle through during "
            "training. If omitted, uses --source_path for all episodes."
        ),
    )
    parser.add_argument(
        "--state_encoder_lr",
        type=float,
        default=None,
        help=(
            "Override pggs_config.state_encoder_lr when --train_state_encoder "
            "is set. E.g. 1e-7 for conservative fine-tuning."
        ),
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    # Resolve context flag
    use_context = not args.no_context

    # Resolve train_state_encoder flag
    if args.train_state_encoder:
        train_se = True
    elif args.no_train_state_encoder:
        train_se = False
    else:
        train_se = False  # default: frozen encoder

    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)

    tb_writer = prepare_output_and_logger(args)

    training_ppo(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        num_episodes=args.num_episodes,
        scene_list=args.scene_list,
        policy_backbone=args.policy_backbone,
        use_context=use_context,
        train_state_encoder_flag=train_se,
        load_state_encoder_path=args.load_state_encoder,
        load_ppo_policy_path=args.load_ppo_policy,
        tb_writer=tb_writer,
        state_encoder_lr_override=args.state_encoder_lr,
        device="cuda",
    )

    print("\nDone.")
