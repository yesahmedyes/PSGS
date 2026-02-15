import torch
from lpipsPyTorch import lpips as lpips_fn
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn


def apply_lr_scaling(
    optimizer: torch.optim.Optimizer, action: torch.Tensor, group_mapping: dict
):
    """
    Apply learning rate scaling to optimizer parameter groups.

    Args:
        optimizer: PyTorch optimizer
        action: LR scaling factors [num_groups] or [batch_size, num_groups]
        group_mapping: Mapping from group name to action index
    """
    # Handle batch dimension if present
    if action.dim() > 1:
        action = action.squeeze(0)

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in group_mapping:
            idx = group_mapping[group_name]
            scale = action[idx].item()

            param_group["rl_scale"] = scale
            base = param_group.get("base_lr", param_group["lr"])
            param_group["lr"] = base * scale


def calculate_lpips_reward(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    net_type: str = "vgg",
    device: str = "cuda",
) -> float:
    """
    Calculate reward based on LPIPS metric over all training views.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance containing training cameras
        render_fn: Rendering function (e.g., from gaussian_renderer.render)
        pipe: Pipeline parameters for rendering
        background: Background color tensor
        net_type: Network type for LPIPS ('vgg', 'alex', or 'squeeze')
        device: Device for computation

    Returns:
        reward: Negative mean LPIPS over all training views.
                Higher reward means better perceptual quality.
    """
    # Get training cameras
    train_cameras = scene.getTrainCameras()

    lpips_values = []

    with torch.no_grad():
        for viewpoint_cam in train_cameras:
            # Get camera pose
            pose = gaussians.get_RT(viewpoint_cam.uid)

            # Render the view
            render_pkg = render_fn(
                viewpoint_cam, gaussians, pipe, background, camera_pose=pose
            )
            rendered_image = render_pkg["render"]

            # Get ground truth image
            gt_image = viewpoint_cam.original_image.to(device)

            # Calculate LPIPS for this view
            lpips_value = lpips_fn(rendered_image, gt_image, net_type=net_type)
            lpips_values.append(lpips_value.item())

    # Calculate mean LPIPS
    mean_lpips = torch.tensor(lpips_values).mean().item()

    # Return negative LPIPS as reward (lower LPIPS = higher reward)
    reward = -mean_lpips

    return reward


def calculate_psnr_reward(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Calculate reward based on PSNR metric over all training views.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance containing training cameras
        render_fn: Rendering function
        pipe: Pipeline parameters for rendering
        background: Background color tensor
        device: Device for computation

    Returns:
        reward: Mean PSNR over all training views (higher is better)
    """
    train_cameras = scene.getTrainCameras()
    psnr_values = []

    with torch.no_grad():
        for viewpoint_cam in train_cameras:
            pose = gaussians.get_RT(viewpoint_cam.uid)
            render_pkg = render_fn(
                viewpoint_cam, gaussians, pipe, background, camera_pose=pose
            )
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint_cam.original_image.to(device), 0.0, 1.0)

            psnr_value = psnr_fn(rendered_image, gt_image).mean()
            psnr_values.append(psnr_value.item())

    mean_psnr = torch.tensor(psnr_values).mean().item()
    return mean_psnr


def calculate_ssim_reward(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Calculate reward based on SSIM metric over all training views.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance containing training cameras
        render_fn: Rendering function
        pipe: Pipeline parameters for rendering
        background: Background color tensor
        device: Device for computation

    Returns:
        reward: Mean SSIM over all training views (higher is better)
    """
    train_cameras = scene.getTrainCameras()
    ssim_values = []

    with torch.no_grad():
        for viewpoint_cam in train_cameras:
            pose = gaussians.get_RT(viewpoint_cam.uid)
            render_pkg = render_fn(
                viewpoint_cam, gaussians, pipe, background, camera_pose=pose
            )
            rendered_image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.to(device)

            ssim_value = ssim_fn(rendered_image, gt_image)
            ssim_values.append(ssim_value.item())

    mean_ssim = torch.tensor(ssim_values).mean().item()
    return mean_ssim


def calculate_combined_reward(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    lpips_weight: float = 1.0,
    psnr_weight: float = 0.1,
    ssim_weight: float = 0.1,
    net_type: str = "vgg",
    device: str = "cuda",
) -> float:
    """
    Calculate combined reward using multiple metrics.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance containing training cameras
        render_fn: Rendering function
        pipe: Pipeline parameters for rendering
        background: Background color tensor
        lpips_weight: Weight for LPIPS (negative, so we minimize it)
        psnr_weight: Weight for PSNR (positive, so we maximize it)
        ssim_weight: Weight for SSIM (positive, so we maximize it)
        net_type: Network type for LPIPS
        device: Device for computation

    Returns:
        reward: Combined weighted reward
    """
    lpips_reward = calculate_lpips_reward(
        gaussians, scene, render_fn, pipe, background, net_type, device
    )
    psnr_reward = calculate_psnr_reward(
        gaussians, scene, render_fn, pipe, background, device
    )
    ssim_reward = calculate_ssim_reward(
        gaussians, scene, render_fn, pipe, background, device
    )

    # Normalize PSNR to similar scale as LPIPS (typically 0-1 range)
    # PSNR is typically 20-40 dB, normalize to 0-1
    psnr_normalized = (psnr_reward - 20.0) / 20.0

    # Combined reward
    combined = (
        lpips_weight * lpips_reward
        + psnr_weight * psnr_normalized
        + ssim_weight * ssim_reward
    )

    return combined


def calculate_reward(
    gaussians,
    scene,
    render_fn,
    pipe,
    background: torch.Tensor,
    reward_type: str = "lpips",
    net_type: str = "vgg",
    lpips_weight: float = 1.0,
    psnr_weight: float = 0.1,
    ssim_weight: float = 0.1,
    reward_scale: float = 1.0,
    device: str = "cuda",
) -> float:
    """
    Unified reward calculation function.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance
        render_fn: Rendering function
        pipe: Pipeline parameters
        background: Background color tensor
        reward_type: Type of reward ('lpips', 'psnr', 'ssim', 'combined')
        net_type: Network type for LPIPS
        lpips_weight: Weight for LPIPS in combined reward
        psnr_weight: Weight for PSNR in combined reward
        ssim_weight: Weight for SSIM in combined reward
        reward_scale: Global scaling factor for reward
        device: Device for computation

    Returns:
        reward: Calculated reward value
    """
    if reward_type == "lpips":
        reward = calculate_lpips_reward(
            gaussians, scene, render_fn, pipe, background, net_type, device
        )
    elif reward_type == "psnr":
        reward = calculate_psnr_reward(
            gaussians, scene, render_fn, pipe, background, device
        )
    elif reward_type == "ssim":
        reward = calculate_ssim_reward(
            gaussians, scene, render_fn, pipe, background, device
        )
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    return reward * reward_scale
