import os

import matplotlib.pyplot as plt
import numpy as np


def find_overlaps(ground_truth, generated):
    # Convert both datasets to integers for consistent comparison
    ground_truth_int = ground_truth.astype(int)
    generated_int = generated.astype(int)

    # Use a set for efficient row matching
    ground_truth_tuples = {tuple(row) for row in ground_truth_int}
    overlaps = np.array([row for row in generated_int if tuple(row) in ground_truth_tuples])

    return overlaps


def plot_2d(ground_truth, generated, filename="scatter_plot_2d_highlighted_overlap.png"):
    """2D scatter plot highlighting overlaps between Ground-Truth and Generated."""
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Find overlapping points
    overlaps = find_overlaps(ground_truth, generated)
    print(f"Number of overlapping points (with duplicates): {overlaps.size}")

    plt.figure()

    # Ground truth points
    plt.scatter(
        ground_truth[:, 0],
        ground_truth[:, 1],
        label="Ground-Truth",
        alpha=0.3,
        s=5,
        color="blue",
        marker="o",
    )

    # Generated points
    plt.scatter(
        generated[:, 0], generated[:, 1], label="Generated", alpha=0.6, s=8, color="red", marker="o"
    )

    # Highlight overlapping points
    if overlaps.size > 0:
        plt.scatter(
            overlaps[:, 0],
            overlaps[:, 1],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    else:
        # Add placeholder for overlapping points if none exist
        plt.scatter(
            [],
            [],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )

    plt.legend()
    plt.title("2D Scatter Plot Highlighting Overlaps")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()


def plot_3d(ground_truth, generated, filename="scatter_plot_3d_highlighted_overlap.png"):
    """3D scatter plot with optimized overlap detection, downsampling, and guaranteed overlap label."""
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    # Find overlapping points
    overlaps = find_overlaps(ground_truth, generated)
    print(f"Number of overlapping points (with duplicates): {overlaps.size}")

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot ground truth points
    ax.scatter(
        ground_truth[:, 0],
        ground_truth[:, 1],
        ground_truth[:, 2],
        label="Ground-Truth",
        alpha=0.1,
        s=5,
        color="blue",
        marker="o",
    )

    # Plot generated points
    ax.scatter(
        generated[:, 0],
        generated[:, 1],
        generated[:, 2],
        label="Generated",
        alpha=0.3,
        s=8,
        color="red",
        marker="o",
    )

    # Highlight overlapping points
    if overlaps.size > 0:
        ax.scatter(
            overlaps[:, 0],
            overlaps[:, 1],
            overlaps[:, 2],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    else:
        # Add a placeholder for overlapping points if there are none
        ax.scatter(
            [],
            [],
            [],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )

    # Plot settings
    ax.set_title("3D Scatter Plot with Overlap Highlighting (Optimized)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Save the plot
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()


def plot_3d_subplots(
    ground_truth, generated, filename="scatter_subplot_3d_highlighted_overlap_subplots.png"
):
    """3D scatter plots: (1) ground truth + overlaps, (2) generated + overlaps."""
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)

    # Find overlapping points
    overlaps = find_overlaps(ground_truth, generated)
    print(f"Number of overlapping points (with duplicates): {overlaps.size}")

    # Create the figure with two subplots
    fig = plt.figure(figsize=(12, 8))

    # First subplot: Ground truth + Overlaps
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        ground_truth[:, 0],
        ground_truth[:, 1],
        ground_truth[:, 2],
        label="Ground-Truth",
        alpha=0.1,
        s=5,
        color="blue",
        marker="o",
    )
    if overlaps.size > 0:
        ax1.scatter(
            overlaps[:, 0],
            overlaps[:, 1],
            overlaps[:, 2],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    else:
        ax1.scatter(
            [],
            [],
            [],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    ax1.set_title("Ground Truth and Overlaps")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Store axis limits from the first subplot
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    z_lim = ax1.get_zlim()

    # Second subplot: Generated + Overlaps
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        generated[:, 0],
        generated[:, 1],
        generated[:, 2],
        label="Generated",
        alpha=0.3,
        s=8,
        color="red",
        marker="o",
    )
    if overlaps.size > 0:
        ax2.scatter(
            overlaps[:, 0],
            overlaps[:, 1],
            overlaps[:, 2],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    else:
        ax2.scatter(
            [],
            [],
            [],
            label="Overlapping Points",
            alpha=1.0,
            s=30,
            color="yellow",
            edgecolor="black",
            marker="o",
        )
    ax2.set_title("Generated and Overlaps")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    # Apply axis limits from the first subplot to the second subplot
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()
