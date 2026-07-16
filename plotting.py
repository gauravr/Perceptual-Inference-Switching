import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings

# Suppress warnings from FFT and empty data slices
warnings.filterwarnings("ignore")


def plot_single_chart(
    ax, df_cell, rel_motion, color, model_avg, theta_deg, is_bottom_row, is_first_col
):
    """
    Plots a single joyplot chart (subplot) containing the subject estimate histogram,
    the computed model line, and vertical reference lines.
    """
    # 1. Plot the Subject Data Histogram[cite: 1]
    if len(df_cell) > 0:
        ax.hist(
            df_cell["relative_estimate"],
            bins=np.linspace(-180, 180, 37),
            density=True,
            color=color,
            alpha=0.8,
        )

    # 2. Plot the computed model[cite: 1]
    ax.plot(theta_deg, model_avg, color="black", lw=1.5)

    # 3. Render reference lines (Prior Mean and Displayed Motion Direction)[cite: 1]
    ax.axvline(0, color="blue", linestyle="--", alpha=0.5, lw=1)
    ax.axvline(rel_motion, color="gray", linestyle="-", alpha=0.5, ymax=0.3, lw=2)

    # 4. Clean up axes for joyplot visualization
    ax.set_yticks([])
    if not is_first_col or not is_bottom_row:
        ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 0.04)
    ax.patch.set_alpha(0.0)  # Transparent backgrounds

    # 5. Handle x-ticks for overlapping effect
    if is_bottom_row:
        ax.set_xticks([-160, -80, 0, 80, 160])
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)


def plot_histograms(data_path, params_path, model_compute, label):
    """
    Reads the behavioral data and constructs the full 36-row joyplot grid,
    delegating the plotting of individual axes to `plot_single_chart`.
    """
    # Read data
    data = pd.read_csv(data_path)
    params = pd.read_csv(params_path)
    output_file = f"output/{label.replace(' ', '_')}_{int(time.time())}.png"

    # Process data: Calculate angles and set relative to the prior mean (0 degrees)
    data["estimate_angle"] = (
        np.degrees(np.arctan2(data["estimate_y"], data["estimate_x"])) % 360
    )
    data["relative_estimate"] = (
        data["estimate_angle"] - data["prior_mean"] + 180
    ) % 360 - 180
    data["relative_motion"] = (
        data["motion_direction"] - data["prior_mean"] + 180
    ) % 360 - 180

    # Filter for 6% coherence as displayed in the original paper figure[cite: 1]
    df_6 = data[data["motion_coherence"] == 0.06]

    prior_stds = [80, 40, 20, 10]
    colors = [
        "#693231",
        "#d14033",
        "#ed962f",
        "#96b340",
    ]  # Brown, Red, Orange, Green[cite: 1]

    # Initialize the joyplot-style grid for 36 rows
    fig, axes = plt.subplots(36, 4, figsize=(12, 28), sharex=True)
    # Tighter spacing for the larger number of rows
    fig.subplots_adjust(hspace=-0.3, wspace=0.1, top=0.95)

    # Evaluate over all 36 absolute directions mapped relative to 225
    motions_to_plot = np.arange(170, -181, -10)
    theta_deg = np.linspace(-180, 180, 361)

    for col_idx, prior_std in enumerate(prior_stds):
        color = colors[col_idx]
        df_prior = df_6[df_6["prior_std"] == prior_std]

        # Add column title with color indication
        axes[0, col_idx].set_title(
            f"Prior Std: {prior_std}°",
            color=color,
            fontsize=14,
            fontweight="bold",
            pad=25,
        )

        for row_idx, rel_motion in enumerate(motions_to_plot):
            ax = axes[row_idx, col_idx]

            # Fetch data segment and compute model
            df_cell = df_prior[df_prior["relative_motion"] == rel_motion]
            model_avg = model_compute(
                theta_deg, rel_motion, prior_std, params, coherence=0.06
            )

            # Formatting flags
            is_bottom_row = row_idx == len(motions_to_plot) - 1
            is_first_col = col_idx == 0

            # Render Single Chart
            plot_single_chart(
                ax,
                df_cell,
                rel_motion,
                color,
                model_avg,
                theta_deg,
                is_bottom_row,
                is_first_col,
            )

    # ---------------------------------------------------------
    # Generate custom proxy artists for the global legend
    # ---------------------------------------------------------
    model_line = mlines.Line2D([], [], color="black", lw=1.5, label=label)
    prior_line = mlines.Line2D(
        [], [], color="blue", linestyle="--", alpha=0.5, lw=1, label="Prior Mean (0°)"
    )
    motion_line = mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="-",
        alpha=0.5,
        lw=2,
        label="Displayed Motion Direction",
    )

    # Add the legend to the top of the entire figure
    fig.legend(
        handles=[model_line, prior_line, motion_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=12,
        frameon=False,
    )

    # Global Axis Labels[cite: 1]
    fig.text(
        0.5,
        0.07,
        "Estimated directions distance relative to the prior mean (°)",
        ha="center",
        fontsize=14,
    )
    fig.text(0.06, 0.5, "Probability", va="center", rotation="vertical", fontsize=14)

    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    print(f"Figure generated and saved as '{output_file}'.")
