import numpy as np
import matplotlib.pyplot as plt
import os

def find_overlaps(ground_truth, generated):
    """Find overlapping points between ground_truth and generated."""
    # Convert both datasets to tuples of integers for hashable row comparison
    ground_truth_tuples = {tuple(row) for row in np.round(ground_truth, decimals=5)}
    generated_tuples = {tuple(row) for row in np.round(generated, decimals=5)}

    # Find overlaps
    overlaps = np.array([row for row in ground_truth_tuples if row in generated_tuples])
    return overlaps

def plot_2d(ground_truth, generated):
    """2D scatter plot highlighting overlaps between Ground-Truth and Generated."""
    output_dir = 'results/'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Find overlapping points
    overlaps = find_overlaps(ground_truth, generated)

    plt.figure()

    # Ground truth points
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label='Ground-Truth',
                alpha=0.3, s=5, color='blue', marker='o')
    
    # Generated points
    plt.scatter(generated[:, 0], generated[:, 1], label='Generated',
                alpha=0.6, s=8, color='red', marker='o')
    
    # Highlight overlapping points
    if overlaps.size > 0:
        plt.scatter(overlaps[:, 0], overlaps[:, 1],
                    label='Overlapping Points', alpha=1.0, s=30, color='yellow', edgecolor='black', marker='o')
    else:
        # Add placeholder for overlapping points if none exist
        plt.scatter([], [], label='Overlapping Points', alpha=1.0, s=30, color='yellow', edgecolor='black', marker='o')

    plt.legend()
    plt.title("2D Scatter Plot Highlighting Overlaps")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'{output_dir}/scatter_plot_2d_highlighted_overlap.png')
    plt.close()

def plot_3d(ground_truth, generated):
    """3D scatter plot with optimized overlap detection, downsampling, and guaranteed overlap label."""
    output_dir = 'results/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find overlapping points
    overlaps = find_overlaps(ground_truth, generated)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth points
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
               label='Ground-Truth', alpha=0.3, s=5, color='blue', marker='o')

    # Plot generated points
    ax.scatter(generated[:, 0], generated[:, 1], generated[:, 2],
               label='Generated', alpha=0.6, s=8, color='red', marker='o')

    # Highlight overlapping points
    if overlaps.size > 0:
        ax.scatter(overlaps[:, 0], overlaps[:, 1], overlaps[:, 2],
                   label='Overlapping Points', alpha=1.0, s=30, color='yellow', edgecolor='black', marker='o')
    else:
        # Add a placeholder for overlapping points if there are none
        ax.scatter([], [], [], label='Overlapping Points', alpha=1.0, s=30, color='yellow', edgecolor='black', marker='o')

    # Plot settings
    ax.set_title("3D Scatter Plot with Overlap Highlighting (Optimized)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Save the plot
    plt.savefig(f'{output_dir}/scatter_plot_3d_highlighted_overlap.png')
    plt.close()