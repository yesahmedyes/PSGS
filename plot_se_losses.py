import argparse
import glob
import os
import matplotlib.pyplot as plt


def parse_loss_file(loss_file_path):
    """
    Parse a state_encoder_losses.txt file.

    Format: iteration,phase_counter,mse_loss

    Returns:
        list of tuples: [(iteration, phase_counter, mse_loss), ...]
    """
    losses = []

    if not os.path.exists(loss_file_path):
        return losses

    with open(loss_file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            try:
                parts = line.split(",")
                iteration = int(parts[0])
                phase_counter = int(parts[1])
                mse_loss = float(parts[2])
                losses.append((iteration, phase_counter, mse_loss))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line '{line}': {e}")
                continue

    return losses


def collect_losses_per_scene(output_dir, scenes, epochs):
    """
    Collect MSE losses for each scene across all epochs.

    Returns:
        dict: {scene_name: [avg_loss_epoch1, avg_loss_epoch2, ...]}
    """
    scene_losses = {}

    for scene in scenes:
        losses_per_epoch = []

        for epoch in range(1, epochs + 1):
            # Find loss file for this scene and epoch
            # Pattern: output_dir/*/{scene}/*/epoch_{epoch}/state_encoder_losses.txt
            pattern = os.path.join(
                output_dir,
                "*",
                scene,
                "*",
                f"epoch_{epoch}",
                "state_encoder_losses.txt",
            )
            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"Warning: No loss file found for scene '{scene}' epoch {epoch}")
                print(f"  Searched pattern: {pattern}")
                losses_per_epoch.append(None)
                continue

            # Use the first matching file (should only be one)
            loss_file = matching_files[0]
            losses = parse_loss_file(loss_file)

            if not losses:
                print(f"Warning: No losses parsed from {loss_file}")
                losses_per_epoch.append(None)
                continue

            # Get the final (average) MSE loss for this epoch
            # Use the last loss value as it represents the final phase
            final_loss = losses[-1][2]  # (iteration, phase_counter, mse_loss)
            losses_per_epoch.append(final_loss)

            print(f"Scene '{scene}' Epoch {epoch}: Final MSE Loss = {final_loss:.6f}")

        scene_losses[scene] = losses_per_epoch

    return scene_losses


def plot_scene_losses(scene_losses, output_dir, plots_dir):
    """
    Create individual plots for each scene showing loss vs epoch.
    """
    os.makedirs(plots_dir, exist_ok=True)

    for scene, losses in scene_losses.items():
        # Filter out None values for plotting
        epochs = []
        valid_losses = []

        for epoch_idx, loss in enumerate(losses, start=1):
            if loss is not None:
                epochs.append(epoch_idx)
                valid_losses.append(loss)

        if not valid_losses:
            print(f"Warning: No valid losses to plot for scene '{scene}'")
            continue

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, valid_losses, marker="o", linewidth=2, markersize=8)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.title(f"SE Training Loss - {scene}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Add value labels on points
        for epoch, loss in zip(epochs, valid_losses):
            plt.annotate(
                f"{loss:.4f}",
                xy=(epoch, loss),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # Save plot
        plot_path = os.path.join(plots_dir, f"{scene}_loss.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Saved plot: {plot_path}")

    # Create a combined plot with all scenes
    plt.figure(figsize=(12, 7))

    for scene, losses in scene_losses.items():
        epochs = []
        valid_losses = []

        for epoch_idx, loss in enumerate(losses, start=1):
            if loss is not None:
                epochs.append(epoch_idx)
                valid_losses.append(loss)

        if valid_losses:
            plt.plot(
                epochs, valid_losses, marker="o", linewidth=2, markersize=6, label=scene
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("SE Training Loss - All Scenes", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    combined_plot_path = os.path.join(plots_dir, "all_scenes_combined.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path, dpi=150)
    plt.close()

    print(f"Saved combined plot: {combined_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot state encoder losses across epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory containing training results",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs trained"
    )
    parser.add_argument(
        "--plots_dir", type=str, required=True, help="Directory to save plots"
    )
    parser.add_argument(
        "--scenes", nargs="+", required=True, help="List of scene names"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("State Encoder Loss Plotting")
    print("=" * 60)
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Scenes: {args.scenes}")
    print(f"Plots Directory: {args.plots_dir}")
    print("=" * 60)

    # Collect losses for all scenes
    print("\nCollecting losses...")
    scene_losses = collect_losses_per_scene(args.output_dir, args.scenes, args.epochs)

    # Generate plots
    print("\nGenerating plots...")
    plot_scene_losses(scene_losses, args.output_dir, args.plots_dir)


if __name__ == "__main__":
    main()
