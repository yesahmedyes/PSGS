#
# Online Decision Transformer training script for PGGS policy learning.
#
# One episode = one full Gaussian training run on one scene.
# The DT policy receives a 771-D state and a return-to-go scalar, and
# outputs 6 LR scaling factors.  Training uses supervised NLL loss on
# trajectories stored in a fixed-size replay buffer.
#

import os
import sys
import uuid
import math
from random import randint
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from argparse import ArgumentParser, Namespace
from torchvision.transforms.functional import resize, normalize

from pggs.config import PGGSConfig
from pggs.odt_policy import OnlineDecisionTransformer
from pggs.ppo_policy import _squash_log_prob, _LOG_MAX_SCALE
from pggs.replay_buffer import DTTransition, TrajectoryReplayBuffer
from pggs.state_encoder import StateEncoder
from pggs.utils import apply_lr_scaling

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

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
# KonIQ++ helpers  (same as train_ppo.py)
# ──────────────────────────────────────────────────────────────────────────────


def _preprocess_image_for_koniq(image_tensor: torch.Tensor) -> torch.Tensor:
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
            koniq_output = koniq_model(preprocessed.unsqueeze(0))
            overall_score = koniq_output[0, 0].item() * k[0] + b[0]
            all_scores.append(overall_score / 100.0)
    return float(np.mean(all_scores))


# ──────────────────────────────────────────────────────────────────────────────
# Episode rollout with ODT
# ──────────────────────────────────────────────────────────────────────────────


def run_episode_dt(
    dataset,
    opt,
    pipe,
    pggs_config: PGGSConfig,
    state_encoder: StateEncoder,
    policy: OnlineDecisionTransformer,
    koniq_model,
    koniq_k,
    koniq_b,
    target_rtg: float,
    train_state_encoder: bool = False,
    device: str = "cuda",
) -> Tuple[List[DTTransition], float]:
    """
    Run one full Gaussian training episode and collect ODT transitions.

    Returns:
        transitions: list of DTTransition namedtuples (rtg filled later by buffer)
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

    print(f"Number of points at initialisation: {gaussians.get_xyz.shape[0]}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # ── Policy state ──────────────────────────────────────────────────────────
    policy.eval()
    state_encoder.eval()
    policy.reset_episode()

    # ── Phase tracking ────────────────────────────────────────────────────────
    transitions: List[DTTransition] = []
    phase_ssim_losses: List[float] = []
    phase_l1_losses: List[float] = []
    views_in_phase = 0
    phase_counter = 0
    timestep = 0
    cumulative_reward = 0.0
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # ── Logging ───────────────────────────────────────────────────────────────
    ema_loss_for_log = 0.0
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    episode_start = time()

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Episode progress")

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
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
                sum(phase_l1_losses) / len(phase_l1_losses) if phase_l1_losses else 0.0
            )

            # ── Encode state ──────────────────────────────────────────────────
            with torch.no_grad():
                state = state_encoder(
                    gaussians=gaussians,
                    iteration=iteration,
                    max_iterations=opt.iterations,
                    avg_ssim_loss=avg_ssim,
                    avg_l1_loss=avg_l1,
                )

            # ── Compute current RTG ───────────────────────────────────────────
            current_rtg = max(target_rtg - cumulative_reward, 0.0)

            # ── Sample action from ODT ────────────────────────────────────────
            action, log_prob = policy.get_action(
                rtg=current_rtg,
                state=state,
                timestep=timestep,
            )

            # ── Apply LR scaling ──────────────────────────────────────────────
            apply_lr_scaling(
                optimizer=gaussians.optimizer,
                action=action,
                group_mapping=pggs_config.group_mapping,
            )

            # ── Reward ────────────────────────────────────────────────────────
            done = iteration == opt.iterations
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

            # ── Append action token to buffer ─────────────────────────────────
            policy.append_action_to_buffer(action)

            cumulative_reward += reward

            # Store raw Gaussian features if encoder will be fine-tuned
            if train_state_encoder:
                gauss_features = (
                    state_encoder.gaussian_encoder._extract_gaussian_features(gaussians)
                    .detach()
                    .cpu()
                )
                ctx = torch.tensor(
                    [min(iteration / opt.iterations, 1.0), avg_ssim, avg_l1],
                    dtype=torch.float32,
                )
            else:
                gauss_features = None
                ctx = None

            transitions.append(
                DTTransition(
                    state=state.detach().cpu(),
                    action=action.detach().cpu(),
                    reward=reward,
                    rtg=current_rtg,  # will be recomputed by replay buffer
                    timestep=timestep,
                    done=done,
                    gaussian_features=gauss_features,
                    context=ctx,
                    iteration=iteration,
                    max_iterations=opt.iterations,
                )
            )

            tqdm.write(
                f"\n[ITER {iteration}] Phase {phase_counter} "
                f"| Avg SSIM loss: {avg_ssim:.6f}  Avg L1 loss: {avg_l1:.6f}"
            )
            tqdm.write(f"  LR scaling: {action.detach().cpu().numpy()}")
            tqdm.write(
                f"  KonIQ++ reward: {reward:.4f}  RTG: {current_rtg:.4f}"
            )

            phase_counter += 1
            timestep += 1
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

        bg = torch.rand((3,), device=device) if opt.random_background else background
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

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            if iteration % 500 == 0 or iteration == opt.iterations:
                torch.cuda.empty_cache()
                train_cameras = scene.getTrainCameras()
                l1_eval = 0.0
                psnr_eval = 0.0
                for cam in train_cameras:
                    cam_pose = gaussians.get_RT(cam.uid)
                    rendered = torch.clamp(
                        render(cam, gaussians, pipe, background, camera_pose=cam_pose)[
                            "render"
                        ],
                        0.0,
                        1.0,
                    )
                    gt = torch.clamp(cam.original_image.to(device), 0.0, 1.0)
                    l1_eval += l1_loss(rendered, gt).mean().double()
                    psnr_eval += psnr(rendered, gt).mean().double()
                l1_eval /= len(train_cameras)
                psnr_eval /= len(train_cameras)
                tqdm.write(
                    f"[ITER {iteration}] Eval train: L1 {l1_eval:.5f}  PSNR {psnr_eval:.2f}"
                )
                torch.cuda.empty_cache()

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()

    episode_time = time() - episode_start
    print(
        f"Episode completed in {episode_time:.1f}s  |  {phase_counter} phases collected"
    )

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
    print(f"Final KonIQ++ reward: {final_reward:.4f}")

    return transitions, final_reward


# ──────────────────────────────────────────────────────────────────────────────
# ODT supervised update
# ──────────────────────────────────────────────────────────────────────────────


def dt_update(
    policy: OnlineDecisionTransformer,
    policy_optimizer: torch.optim.Optimizer,
    replay_buffer: TrajectoryReplayBuffer,
    pggs_config: PGGSConfig,
    device: str = "cuda",
) -> dict:
    """
    Perform supervised ODT updates on the replay buffer.

    Loss = NLL of actual actions under the DT's predicted distribution
           - entropy_coeff * entropy.

    Returns dict of average loss values for logging.
    """
    policy.train()

    total_nll = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = pggs_config.odt_updates_per_episode

    for _ in range(n_updates):
        # Sample batch
        if pggs_config.odt_use_weighted_sampling:
            batch = replay_buffer.sample_batch_weighted(
                pggs_config.odt_batch_size, pggs_config.odt_context_len
            )
        else:
            batch = replay_buffer.sample_batch(
                pggs_config.odt_batch_size, pggs_config.odt_context_len
            )

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rtgs = batch["rtgs"].to(device)
        timesteps = batch["timesteps"].to(device)
        mask = batch["attention_mask"].to(device)

        # Forward
        action_means, action_log_std = policy(
            rtgs=rtgs,
            states=states,
            actions=actions,
            timesteps=timesteps,
            attention_mask=mask,
        )

        # Compute NLL with squashing correction
        c = _LOG_MAX_SCALE
        log_a = torch.log(actions.clamp(min=1e-6))
        raw_actions = torch.atanh((log_a / c).clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        std = torch.exp(action_log_std.clamp(-5, 2))
        dist = Normal(action_means, std)
        log_prob_raw = dist.log_prob(raw_actions).sum(-1)  # [B, T]

        # Jacobian correction per-timestep
        log_jac = (
            c * torch.tanh(raw_actions)
            + math.log(c)
            + torch.log(1.0 - torch.tanh(raw_actions) ** 2 + 1e-6)
        )
        log_probs = log_prob_raw - log_jac.sum(dim=-1)  # [B, T]

        # Mask padding
        mask_float = mask.float()
        valid_count = mask_float.sum().clamp(min=1.0)

        nll_loss = -(log_probs * mask_float).sum() / valid_count

        # Entropy bonus
        entropy = (dist.entropy().sum(-1) * mask_float).sum() / valid_count

        loss = nll_loss - pggs_config.odt_entropy_coeff * entropy

        policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), pggs_config.odt_gradient_clip)
        policy_optimizer.step()

        total_nll += nll_loss.item()
        total_entropy += entropy.item()
        total_loss += loss.item()

    n = max(n_updates, 1)
    return {
        "nll_loss": total_nll / n,
        "entropy": total_entropy / n,
        "total_loss": total_loss / n,
    }


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


def training_dt(
    dataset,
    opt,
    pipe,
    use_context: bool,
    train_state_encoder_flag: bool,
    load_state_encoder_path: Optional[str],
    load_odt_policy_path: Optional[str],
    load_replay_buffer_path: Optional[str],
    log_dir: str,
    episode: int,
    epoch: int,
    tb_writer,
    target_rtg_override: Optional[float] = None,
    state_encoder_lr_override: Optional[float] = None,
    save_state_encoder_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Run exactly one ODT episode for the scene described by ``dataset``.

    The shell script drives all loops (epoch / n_views / gs_train_iter / scene).
    This function:
        1. Loads policy + state encoder + replay buffer from checkpoints
        2. Runs one episode and collects trajectory
        3. Adds trajectory to replay buffer
        4. Performs supervised ODT updates (if buffer has enough episodes)
        5. Saves checkpoints
    """
    pggs_config = PGGSConfig()
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
        ckpt = torch.load(
            load_state_encoder_path, map_location=device, weights_only=False
        )
        state_key = "state_encoder" if "state_encoder" in ckpt else None
        state_encoder.load_state_dict(ckpt[state_key] if state_key else ckpt)
        print("State encoder loaded.")
    else:
        if load_state_encoder_path:
            print(
                f"Warning: state encoder checkpoint not found at {load_state_encoder_path}"
            )
        print("Initialising state encoder from scratch.")

    if not train_state_encoder_flag:
        state_encoder.eval()
        for p in state_encoder.parameters():
            p.requires_grad_(False)

    state_dim = state_encoder.get_output_dim()

    # ── ODT Policy ────────────────────────────────────────────────────────────
    policy = OnlineDecisionTransformer(
        state_dim=state_dim,
        action_dim=pggs_config.num_lr_params,
        hidden_dim=pggs_config.odt_hidden_dim,
        n_heads=pggs_config.odt_n_heads,
        n_layers=pggs_config.odt_n_layers,
        ffn_dim=pggs_config.odt_ffn_dim,
        dropout=pggs_config.odt_dropout,
        max_episode_len=pggs_config.odt_max_episode_len,
        device=device,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=pggs_config.odt_lr,
        weight_decay=pggs_config.odt_weight_decay,
    )

    # Load policy checkpoint
    if load_odt_policy_path and os.path.exists(load_odt_policy_path):
        print(f"Loading ODT policy from {load_odt_policy_path}")
        ckpt = torch.load(load_odt_policy_path, map_location=device, weights_only=False)
        policy_key = "policy" if "policy" in ckpt else None
        policy.load_state_dict(ckpt[policy_key] if policy_key else ckpt)
        if "optimizer" in ckpt:
            policy_optimizer.load_state_dict(ckpt["optimizer"])
        print("ODT policy loaded.")
    else:
        if load_odt_policy_path:
            print(f"Warning: ODT policy checkpoint not found at {load_odt_policy_path}")
        print("Initialising ODT policy from scratch.")

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    replay_buffer = TrajectoryReplayBuffer(
        max_episodes=pggs_config.odt_replay_buffer_size,
        discount=pggs_config.odt_rtg_discount,
    )
    if load_replay_buffer_path and os.path.exists(load_replay_buffer_path):
        print(f"Loading replay buffer from {load_replay_buffer_path}")
        replay_buffer = torch.load(
            load_replay_buffer_path, map_location="cpu", weights_only=False
        )
        print(f"Replay buffer loaded ({len(replay_buffer)} episodes).")
    else:
        if load_replay_buffer_path:
            print(
                f"Warning: replay buffer not found at {load_replay_buffer_path}"
            )
        print("Starting with empty replay buffer.")

    # ── Determine target RTG ──────────────────────────────────────────────────
    if target_rtg_override is not None:
        target_rtg = target_rtg_override
    elif pggs_config.odt_rtg_adapt and len(replay_buffer) > 0:
        target_rtg = (
            replay_buffer.get_best_return() * pggs_config.odt_rtg_adapt_factor
        )
        target_rtg = max(target_rtg, pggs_config.odt_target_rtg)
    else:
        target_rtg = pggs_config.odt_target_rtg

    print(f"Target RTG for this episode: {target_rtg:.4f}")

    # ── KonIQ++ model ─────────────────────────────────────────────────────────
    from koniqplusplus.IQAmodel import Model_Joint

    print(f"Loading KonIQ++ model from {pggs_config.koniq_model_path} ...")
    koniq_model = Model_Joint().to(device)
    koniq_ckpt = torch.load(
        pggs_config.koniq_model_path, map_location=device, weights_only=False
    )
    koniq_model.load_state_dict(koniq_ckpt["model"])
    koniq_k = koniq_ckpt["k"]
    koniq_b = koniq_ckpt["b"]
    koniq_model.eval()
    print("KonIQ++ loaded.")

    # ── Run episode ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Episode {episode}  Epoch {epoch}  scene: {dataset.source_path}")
    print(f"{'=' * 60}")

    transitions, final_reward = run_episode_dt(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        pggs_config=pggs_config,
        state_encoder=state_encoder,
        policy=policy,
        koniq_model=koniq_model,
        koniq_k=koniq_k,
        koniq_b=koniq_b,
        target_rtg=target_rtg,
        train_state_encoder=train_state_encoder_flag,
        device=device,
    )

    print(
        f"  {len(transitions)} phases collected, final KonIQ++ reward = {final_reward:.4f}"
    )

    # ── Add to replay buffer ──────────────────────────────────────────────────
    replay_buffer.add_episode(transitions)
    print(
        f"  Replay buffer: {len(replay_buffer)} episodes, "
        f"best return: {replay_buffer.get_best_return():.4f}, "
        f"mean return: {replay_buffer.get_mean_return():.4f}"
    )

    # ── ODT update ────────────────────────────────────────────────────────────
    losses = {}
    if len(replay_buffer) >= pggs_config.odt_warmup_episodes:
        losses = dt_update(
            policy=policy,
            policy_optimizer=policy_optimizer,
            replay_buffer=replay_buffer,
            pggs_config=pggs_config,
            device=device,
        )
        print(
            f"  ODT update — nll_loss: {losses.get('nll_loss', 0):.4f}  "
            f"entropy: {losses.get('entropy', 0):.4f}  "
            f"total_loss: {losses.get('total_loss', 0):.4f}"
        )
    else:
        print(
            f"  Skipping ODT update (need {pggs_config.odt_warmup_episodes} episodes, "
            f"have {len(replay_buffer)})"
        )

    # ── Logging ───────────────────────────────────────────────────────────────
    if tb_writer:
        tb_writer.add_scalar("odt/final_reward", final_reward, episode)
        tb_writer.add_scalar("odt/target_rtg", target_rtg, episode)
        tb_writer.add_scalar("odt/n_phases", len(transitions), episode)
        tb_writer.add_scalar("odt/buffer_size", len(replay_buffer), episode)
        tb_writer.add_scalar(
            "odt/best_return", replay_buffer.get_best_return(), episode
        )
        for k, v in losses.items():
            tb_writer.add_scalar(f"odt/{k}", v, episode)

    # ── CSV logging ───────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "dt_losses.csv")
    write_header = not os.path.exists(csv_path)
    src_parts = os.path.normpath(dataset.source_path).split(os.sep)
    scene_label = "/".join(src_parts[-2:]) if len(src_parts) >= 2 else src_parts[-1]
    with open(csv_path, "a") as _f:
        if write_header:
            _f.write(
                "episode,epoch,scene,nll_loss,entropy,total_loss,"
                "final_reward,target_rtg,buffer_size,best_return\n"
            )
        _f.write(
            f"{episode},{epoch},{scene_label},"
            f"{losses.get('nll_loss', float('nan')):.6f},"
            f"{losses.get('entropy', float('nan')):.6f},"
            f"{losses.get('total_loss', float('nan')):.6f},"
            f"{final_reward:.6f},{target_rtg:.6f},"
            f"{len(replay_buffer)},{replay_buffer.get_best_return():.6f}\n"
        )

    # ── Save checkpoints ──────────────────────────────────────────────────────
    _save_checkpoints(
        policy=policy,
        state_encoder=state_encoder,
        policy_optimizer=policy_optimizer,
        replay_buffer=replay_buffer,
        episode=episode,
        pggs_config=pggs_config,
        train_se=train_state_encoder_flag,
        save_state_encoder_path=save_state_encoder_path,
    )

    print("\nEpisode complete.")


def _save_checkpoints(
    policy: OnlineDecisionTransformer,
    state_encoder: StateEncoder,
    policy_optimizer,
    replay_buffer: TrajectoryReplayBuffer,
    episode: int,
    pggs_config: PGGSConfig,
    train_se: bool,
    save_state_encoder_path: Optional[str] = None,
):
    os.makedirs("checkpoints", exist_ok=True)

    # Policy
    policy_ckpt = {
        "policy": policy.state_dict(),
        "optimizer": policy_optimizer.state_dict(),
        "episode": episode,
        "hidden_dim": pggs_config.odt_hidden_dim,
        "action_dim": pggs_config.num_lr_params,
    }
    torch.save(policy_ckpt, pggs_config.odt_policy_checkpoint)
    print(f"  ODT policy saved → {pggs_config.odt_policy_checkpoint}")

    # Replay buffer
    torch.save(replay_buffer, pggs_config.odt_replay_buffer_checkpoint)
    print(f"  Replay buffer saved → {pggs_config.odt_replay_buffer_checkpoint}")

    # State encoder (if fine-tuning)
    if train_se:
        se_save_path = save_state_encoder_path or pggs_config.state_encoder_checkpoint
        se_ckpt = {
            "state_encoder": state_encoder.state_dict(),
            "episode": episode,
            "use_context": pggs_config.use_context,
        }
        os.makedirs(os.path.dirname(se_save_path) or ".", exist_ok=True)
        torch.save(se_ckpt, se_save_path)
        print(f"  State encoder saved → {se_save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Online Decision Transformer training for PGGS")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

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
        help="Exclude context scalars",
    )
    parser.add_argument(
        "--train_state_encoder",
        action="store_true",
        default=False,
        help="Fine-tune state encoder during ODT training",
    )
    parser.add_argument(
        "--load_state_encoder",
        type=str,
        default=None,
        help="Path to pre-trained state encoder checkpoint",
    )
    parser.add_argument(
        "--save_state_encoder",
        type=str,
        default=None,
        help="Path to save state encoder checkpoint",
    )
    parser.add_argument(
        "--load_odt_policy",
        type=str,
        default=None,
        help="Path to pre-trained ODT policy checkpoint",
    )
    parser.add_argument(
        "--load_replay_buffer",
        type=str,
        default=None,
        help="Path to saved replay buffer checkpoint",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Shared output directory for dt_losses.csv and TensorBoard logs",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Global episode counter (set by shell script)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Current epoch number (set by shell script)",
    )
    parser.add_argument(
        "--target_rtg",
        type=float,
        default=None,
        help="Override target RTG for this episode (default: adaptive or config)",
    )
    parser.add_argument(
        "--state_encoder_lr",
        type=float,
        default=None,
        help="Override state encoder learning rate",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    use_context = not args.no_context

    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)

    tb_writer = prepare_output_and_logger(args)

    training_dt(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        use_context=use_context,
        train_state_encoder_flag=args.train_state_encoder,
        load_state_encoder_path=args.load_state_encoder,
        load_odt_policy_path=args.load_odt_policy,
        load_replay_buffer_path=args.load_replay_buffer,
        log_dir=args.log_dir,
        episode=args.episode,
        epoch=args.epoch,
        tb_writer=tb_writer,
        target_rtg_override=args.target_rtg,
        state_encoder_lr_override=args.state_encoder_lr,
        save_state_encoder_path=args.save_state_encoder,
        device="cuda",
    )

    print("\nDone.")
