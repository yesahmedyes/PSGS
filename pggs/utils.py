import torch
from lpipsPyTorch import lpips as lpips_fn


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
