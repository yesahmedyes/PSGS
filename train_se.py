#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from random import randint
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from torchvision.transforms.functional import resize, normalize

from pggs.config import PGGSConfig

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor
from utils.sfm_utils import save_time
import uuid

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]

    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)

    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device="cuda", scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.

    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.

    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale

    return lr_modifiers


class RewardPredictionHead(nn.Module):
    def __init__(self, state_dim=771, hidden_dim1=256, hidden_dim2=128, output_dim=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, state):
        """
        Args:
            state: Encoded state tensor [state_dim] or [batch, state_dim]
        Returns:
            predictions: [5] or [batch, 5] tensor of predicted scores
        """
        return self.network(state)


def preprocess_image_for_koniq(image_tensor):
    """
    Preprocess rendered image for KonIQ++ model.

    Args:
        image_tensor: [3, H, W] tensor in range [0, 1]

    Returns:
        preprocessed: [3, 480, 640] tensor normalized with ImageNet stats
    """
    # Resize to KonIQ++ expected size
    image_resized = resize(image_tensor, (480, 640))

    # Normalize with ImageNet stats
    image_normalized = normalize(
        image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return image_normalized


def evaluate_with_koniq(
    gaussians, scene, render_fn, pipe, background, koniq_model, k, b, device="cuda"
):
    """
    Render all training views and compute averaged KonIQ++ scores.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance
        render_fn: Render function
        pipe: Pipeline parameters
        background: Background color tensor
        koniq_model: Pretrained KonIQ++ model
        k: Scaling parameter from KonIQ++ checkpoint
        b: Bias parameter from KonIQ++ checkpoint
        device: Device for computation

    Returns:
        averaged_targets: [5] tensor containing averaged transformed scores
            [overall_score, 1-artifacts, 1-blur, 1-contrast, 1-color]
    """
    train_cameras = scene.getTrainCameras()
    all_scores = []

    with torch.no_grad():
        for viewpoint_cam in train_cameras:
            # Get camera pose
            pose = gaussians.get_RT(viewpoint_cam.uid)

            # Render the view
            render_pkg = render_fn(
                viewpoint_cam, gaussians, pipe, background, camera_pose=pose
            )
            rendered_image = render_pkg["render"]  # [3, H, W]

            # Clamp to [0, 1]
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

            # Preprocess for KonIQ++
            preprocessed_image = preprocess_image_for_koniq(rendered_image)

            # Pass through KonIQ++ model
            koniq_output = koniq_model(preprocessed_image.unsqueeze(0))  # [1, 5]

            # Extract scores: [overall, artifacts, blur, contrast, color]
            overall_score = koniq_output[0, 0].item() * k[0] + b[0]
            defect_scores = koniq_output[0, 1:].cpu().numpy()  # [4]

            # Transform: overall_score stays as is, defect scores become (1 - defect)
            transformed_scores = [
                overall_score,
                1.0 - defect_scores[0],  # 1 - artifacts
                1.0 - defect_scores[1],  # 1 - blur
                1.0 - defect_scores[2],  # 1 - contrast
                1.0 - defect_scores[3],  # 1 - color
            ]

            all_scores.append(transformed_scores)

    # Average across all views
    all_scores = np.array(all_scores)  # [num_views, 5]

    averaged_targets = torch.tensor(
        all_scores.mean(axis=0), dtype=torch.float32, device=device
    )

    return averaged_targets


def train_state_encoder_step(
    state_encoder,
    prediction_head,
    optimizer,
    gaussians,
    iteration,
    max_iterations,
    avg_ssim_loss,
    avg_l1_loss,
    targets,
    device="cuda",
):
    """
    Perform one training step for the state encoder and prediction head.

    Args:
        state_encoder: StateEncoder instance
        prediction_head: RewardPredictionHead instance
        optimizer: Optimizer for state encoder + prediction head
        gaussians: GaussianModel instance
        iteration: Current iteration
        max_iterations: Maximum iterations
        avg_ssim_loss: Average SSIM loss for current phase
        avg_l1_loss: Average L1 loss for current phase
        targets: [5] tensor of target scores from KonIQ++
        device: Device for computation

    Returns:
        loss: MSE loss value
        predictions: [5] tensor of predicted scores
    """
    # Set models to training mode
    state_encoder.train()
    prediction_head.train()

    # Encode current state
    state = state_encoder(
        gaussians=gaussians,
        iteration=iteration,
        max_iterations=max_iterations,
        avg_ssim_loss=avg_ssim_loss,
        avg_l1_loss=avg_l1_loss,
    )  # [771]

    # Predict scores
    predictions = prediction_head(state)  # [5]

    # Compute MSE loss
    loss = nn.functional.mse_loss(predictions, targets)

    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), predictions.detach()


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    load_state_encoder_path=None,
    train_state_encoder_override=None,
):
    pggs_config = PGGSConfig()
    
    # Apply command line override if provided
    if train_state_encoder_override is not None:
        pggs_config.train_state_encoder = train_state_encoder_override
        print(f"State encoder training {'enabled' if train_state_encoder_override else 'disabled'} via command line")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # per-point-optimizer
    confidence_path = os.path.join(
        dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy"
    )
    confidence_lr = load_and_prepare_confidence(
        confidence_path, device="cuda", scale=(1, 100)
    )
    scene = Scene(dataset, gaussians)

    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)
    else:
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    train_cams_init = scene.getTrainCameras().copy()

    for save_iter in saving_iterations:
        os.makedirs(scene.model_path + f"/pose/ours_{save_iter}", exist_ok=True)
        save_pose(
            scene.model_path + f"/pose/ours_{save_iter}/pose_org.npy",
            gaussians.P,
            train_cams_init,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize state encoder training components
    state_encoder = None
    prediction_head = None
    state_encoder_optimizer = None
    koniq_model = None
    koniq_k = None
    koniq_b = None
    phase_counter = 0
    views_in_current_phase = 0
    phase_ssim_losses = []
    phase_l1_losses = []

    if pggs_config.train_state_encoder:
        print("Initializing State Encoder Training...")

        from pggs.state_encoder import StateEncoder
        from koniqplusplus.IQAmodel import Model_Joint

        # Initialize state encoder
        state_encoder = StateEncoder(
            num_inducing_vectors=pggs_config.num_inducing_vectors,
            d_model=pggs_config.state_d_model,
            num_heads=pggs_config.state_num_heads,
            dropout=pggs_config.state_dropout,
            sh_degree=dataset.sh_degree,
        ).to("cuda")

        # Initialize prediction head
        prediction_head = RewardPredictionHead(
            state_dim=state_encoder.get_output_dim(),
            hidden_dim1=256,
            hidden_dim2=128,
            output_dim=5,
        ).to("cuda")

        # Initialize optimizer for state encoder + prediction head
        state_encoder_optimizer = torch.optim.Adam(
            list(state_encoder.parameters()) + list(prediction_head.parameters()),
            lr=pggs_config.reward_prediction_lr,
        )
        
        # Load state encoder checkpoint if provided
        if load_state_encoder_path and os.path.exists(load_state_encoder_path):
            print(f"Loading state encoder checkpoint from {load_state_encoder_path}...")
            se_checkpoint = torch.load(load_state_encoder_path, map_location="cuda")
            state_encoder.load_state_dict(se_checkpoint['state_encoder'])
            prediction_head.load_state_dict(se_checkpoint['prediction_head'])
            if 'optimizer' in se_checkpoint:
                state_encoder_optimizer.load_state_dict(se_checkpoint['optimizer'])
            if 'phase_counter' in se_checkpoint:
                phase_counter = se_checkpoint['phase_counter']
                print(f"Resuming from phase {phase_counter}")
            print("State encoder checkpoint loaded successfully!")
        elif load_state_encoder_path:
            print(f"Warning: State encoder checkpoint not found at {load_state_encoder_path}")
            print("Starting with random initialization")

        # Load pretrained KonIQ++ model
        print(f"Loading KonIQ++ model from {pggs_config.koniq_model_path}...")
        koniq_model = Model_Joint().to("cuda")
        koniq_checkpoint = torch.load(pggs_config.koniq_model_path, map_location="cuda")
        koniq_model.load_state_dict(koniq_checkpoint["model"])
        koniq_k = koniq_checkpoint["k"]
        koniq_b = koniq_checkpoint["b"]
        koniq_model.eval()  # Keep in eval mode

        print("State encoder training initialized successfully!")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    start = time()
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if opt.optim_pose == False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # State Encoder Training: Check if phase is complete
        if (
            pggs_config.train_state_encoder
            and iteration >= pggs_config.start_iteration
            and views_in_current_phase >= pggs_config.phase_length
        ):
            print(
                f"\n[ITER {iteration}]: Training state encoder (phase {phase_counter})"
            )

            # Calculate average losses for the completed phase
            avg_ssim_loss = (
                sum(phase_ssim_losses) / len(phase_ssim_losses)
                if phase_ssim_losses
                else 0.0
            )
            avg_l1_loss = (
                sum(phase_l1_losses) / len(phase_l1_losses) if phase_l1_losses else 0.0
            )

            # Evaluate with KonIQ++ to get target scores
            targets = evaluate_with_koniq(
                gaussians=gaussians,
                scene=scene,
                render_fn=render,
                pipe=pipe,
                background=background,
                koniq_model=koniq_model,
                k=koniq_k,
                b=koniq_b,
                device="cuda",
            )

            # Train state encoder
            se_loss, predictions = train_state_encoder_step(
                state_encoder=state_encoder,
                prediction_head=prediction_head,
                optimizer=state_encoder_optimizer,
                gaussians=gaussians,
                iteration=iteration,
                max_iterations=opt.iterations,
                avg_ssim_loss=avg_ssim_loss,
                avg_l1_loss=avg_l1_loss,
                targets=targets,
                device="cuda",
            )

            # Log state encoder training
            print(f"  Phase {phase_counter}")
            print(f"  MSE Loss: {se_loss:.6f}")
            print(f"  Predictions: {predictions.cpu().numpy()}")
            print(f"  Targets: {targets.cpu().numpy()}")

            # Tensorboard logging
            if tb_writer:
                tb_writer.add_scalar("state_encoder/mse_loss", se_loss, iteration)
                tb_writer.add_scalar(
                    "state_encoder/overall_score_error",
                    abs(predictions[0].item() - targets[0].item()),
                    iteration,
                )
                avg_defect_error = (
                    torch.abs(predictions[1:] - targets[1:]).mean().item()
                )
                tb_writer.add_scalar(
                    "state_encoder/avg_defect_error", avg_defect_error, iteration
                )
                tb_writer.add_scalar(
                    "state_encoder/phase_counter", phase_counter, iteration
                )

                # Log individual predictions and targets
                pred_names = [
                    "overall_score",
                    "1-artifacts",
                    "1-blur",
                    "1-contrast",
                    "1-color",
                ]
                for idx, name in enumerate(pred_names):
                    tb_writer.add_scalar(
                        f"state_encoder/prediction_{name}",
                        predictions[idx].item(),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"state_encoder/target_{name}", targets[idx].item(), iteration
                    )
            
            # Log MSE loss to file for plotting
            loss_log_path = os.path.join(scene.model_path, "state_encoder_losses.txt")
            with open(loss_log_path, "a") as f:
                f.write(f"{iteration},{phase_counter},{se_loss}\n")

            # Reset phase tracking
            phase_counter += 1
            views_in_current_phase = 0
            phase_ssim_losses = []
            phase_l1_losses = []

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # State Encoder: Track losses for current phase
            if pggs_config.train_state_encoder:
                phase_ssim_losses.append((1.0 - ssim_value).item())
                phase_l1_losses.append(Ll1.item())
                views_in_current_phase += 1

            # Densification
            # if iteration < opt.densify_until_iter:
            # # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Log and save
            if iteration == opt.iterations:
                end = time()
                train_time_wo_log = end - start
                save_time(
                    scene.model_path, "[2] train_joint_TrainTime", train_time_wo_log
                )
                training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(
                    scene.model_path + f"/pose/ours_{iteration}/pose_optimized.npy",
                    gaussians.P,
                    train_cams_init,
                )

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    end = time()
    train_time = end - start
    save_time(scene.model_path, "[2] train_joint", train_time)

    # Save state encoder checkpoint after training completes
    if pggs_config.train_state_encoder:
        state_encoder_checkpoint = {
            "state_encoder": state_encoder.state_dict(),
            "prediction_head": prediction_head.state_dict(),
            "optimizer": state_encoder_optimizer.state_dict(),
            "phase_counter": phase_counter,
            "config": {
                "state_dim": state_encoder.get_output_dim(),
                "num_inducing_vectors": pggs_config.num_inducing_vectors,
                "state_d_model": pggs_config.state_d_model,
                "state_num_heads": pggs_config.state_num_heads,
                "state_dropout": pggs_config.state_dropout,
                "sh_degree": dataset.sh_degree,
            },
        }

        # Save to pggs folder
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "state_encoder_final.pth")

        torch.save(state_encoder_checkpoint, checkpoint_path)

        print(f"State encoder saved to: {checkpoint_path}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(len(scene.getTrainCameras()))
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    if config["name"] == "train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(
                        renderFunc(
                            viewpoint, scene.gaussians, *renderArgs, camera_pose=pose
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_state_encoder", type=str, default=None, help="Path to state encoder checkpoint to load")
    parser.add_argument("--train_state_encoder", action="store_true", help="Enable state encoder training (overrides config)")
    parser.add_argument("--no_train_state_encoder", action="store_true", help="Disable state encoder training (overrides config)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Determine if state encoder training should be enabled
    train_se_enabled = None
    if args.train_state_encoder:
        train_se_enabled = True
    elif args.no_train_state_encoder:
        train_se_enabled = False
    
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.load_state_encoder,
        train_se_enabled,
    )

    # All done
    print("\nTraining complete.")
