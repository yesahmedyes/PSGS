import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_ppo_csv(csv_path):
    """
    Parse ppo_losses.csv produced by train_ppo.py.

    Expected header:
        episode,epoch,scene,policy_loss,value_loss,entropy,final_reward

    Returns:
        list of dicts, one per row.
    """
    rows = []
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return rows

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            try:
                rows.append(
                    {
                        "episode": int(line["episode"]),
                        "epoch": int(line["epoch"]),
                        "scene": line["scene"].strip(),
                        "policy_loss": float(line["policy_loss"]),
                        "value_loss": float(line["value_loss"]),
                        "entropy": float(line["entropy"]),
                        "final_reward": float(line["final_reward"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse row {line}: {e}")
    return rows


def aggregate_by_scene_epoch(rows):
    """
    Group rows by (scene, epoch) and average numeric metrics.

    Returns:
        dict: {scene: {epoch: {metric: avg_value, ...}}}
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row["scene"]][row["epoch"]].append(row)

    aggregated = {}
    for scene, epoch_map in buckets.items():
        aggregated[scene] = {}
        for epoch, episode_rows in epoch_map.items():
            aggregated[scene][epoch] = {
                "policy_loss": sum(r["policy_loss"] for r in episode_rows) / len(episode_rows),
                "value_loss": sum(r["value_loss"] for r in episode_rows) / len(episode_rows),
                "entropy": sum(r["entropy"] for r in episode_rows) / len(episode_rows),
                "final_reward": sum(r["final_reward"] for r in episode_rows) / len(episode_rows),
            }
    return aggregated


def plot_scene(scene, epoch_data, plots_dir, n_views=None):
    """Per-scene plot: reward (left y-axis) + policy_loss (right y-axis) vs epoch."""
    epochs = sorted(epoch_data.keys())
    rewards = [epoch_data[e]["final_reward"] for e in epochs]
    policy_losses = [epoch_data[e]["policy_loss"] for e in epochs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_reward = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Final Reward", color=color_reward, fontsize=12)
    ax1.plot(epochs, rewards, marker="o", linewidth=2, markersize=6,
             color=color_reward, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color_reward)

    ax2 = ax1.twinx()
    color_loss = "tab:red"
    ax2.set_ylabel("Policy Loss", color=color_loss, fontsize=12)
    ax2.plot(epochs, policy_losses, marker="s", linewidth=2, markersize=6,
             linestyle="--", color=color_loss, label="Policy Loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    title = f"PPO Training — {scene}"
    if n_views is not None:
        title += f"  ({n_views} views)"
    plt.title(title, fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="best")

    ax1.grid(True, alpha=0.3)

    safe_name = scene.replace("/", "_").replace(":", "_")
    plot_path = os.path.join(plots_dir, f"{safe_name}_ppo.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_combined_reward(aggregated, plots_dir, n_views=None):
    """Combined plot: reward vs epoch for all scenes."""
    plt.figure(figsize=(12, 7))

    for scene, epoch_data in aggregated.items():
        epochs = sorted(epoch_data.keys())
        rewards = [epoch_data[e]["final_reward"] for e in epochs]
        plt.plot(epochs, rewards, marker="o", linewidth=2, markersize=6, label=scene)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Final Reward", fontsize=12)
    title = "PPO Training — Reward (All Scenes)"
    if n_views is not None:
        title += f"  ({n_views} views)"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "all_scenes_reward.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved combined reward plot: {path}")


def plot_combined_policy_loss(aggregated, plots_dir, n_views=None):
    """Combined plot: policy loss vs epoch for all scenes."""
    plt.figure(figsize=(12, 7))

    for scene, epoch_data in aggregated.items():
        epochs = sorted(epoch_data.keys())
        losses = [epoch_data[e]["policy_loss"] for e in epochs]
        plt.plot(epochs, losses, marker="s", linewidth=2, markersize=6,
                 linestyle="--", label=scene)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Policy Loss", fontsize=12)
    title = "PPO Training — Policy Loss (All Scenes)"
    if n_views is not None:
        title += f"  ({n_views} views)"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "all_scenes_policy_loss.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved combined policy-loss plot: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot PPO training metrics (reward + policy loss) per epoch per scene"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing ppo_losses.csv (the --model_path used for train_ppo.py)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Total number of epochs trained (used for display / axis reference)",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        required=True,
        help="Directory to save generated .png plots",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Optional list of scene labels to filter (e.g. tanks_templates/horse). "
            "If omitted, all scenes found in the CSV are plotted."
        ),
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=None,
        help="n_views value for this run (used in plot titles only)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PPO Loss Plotting")
    print("=" * 60)
    print(f"Output Directory : {args.output_dir}")
    print(f"Epochs           : {args.epochs}")
    print(f"Plots Directory  : {args.plots_dir}")
    print(f"Scenes filter    : {args.scenes}")
    print(f"N-views          : {args.n_views}")
    print("=" * 60)

    csv_path = os.path.join(args.output_dir, "ppo_losses.csv")
    rows = parse_ppo_csv(csv_path)

    if not rows:
        print("No data found. Exiting.")
        return

    aggregated = aggregate_by_scene_epoch(rows)

    # Optional scene filter
    if args.scenes:
        # Normalise separators for comparison
        def _normalise(s):
            return s.replace(":", "/").replace("\\", "/")

        filter_set = {_normalise(s) for s in args.scenes}
        aggregated = {
            scene: data
            for scene, data in aggregated.items()
            if _normalise(scene) in filter_set
        }
        if not aggregated:
            print(
                "Warning: Scene filter matched no entries in the CSV. "
                "Check the --scenes labels against the CSV content."
            )
            return

    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\nGenerating plots for {len(aggregated)} scene(s)...")
    for scene, epoch_data in aggregated.items():
        print(f"  Scene: {scene}  ({len(epoch_data)} epoch(s))")
        for epoch, metrics in sorted(epoch_data.items()):
            print(
                f"    Epoch {epoch:3d}: reward={metrics['final_reward']:.4f}  "
                f"policy_loss={metrics['policy_loss']:.4f}  "
                f"value_loss={metrics['value_loss']:.4f}  "
                f"entropy={metrics['entropy']:.4f}"
            )
        plot_scene(scene, epoch_data, args.plots_dir, n_views=args.n_views)

    plot_combined_reward(aggregated, args.plots_dir, n_views=args.n_views)
    plot_combined_policy_loss(aggregated, args.plots_dir, n_views=args.n_views)

    print("\nDone.")


if __name__ == "__main__":
    main()
