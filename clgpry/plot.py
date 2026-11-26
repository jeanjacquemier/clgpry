import matplotlib.pyplot as plt
import numpy as np


def save_loglike_histogram(
    loglike_values_path, output_path="loglike_histogram.png", bins=50
):
    """Generate and save a histogram of loglike values from a .npy file.

    Args:
        loglike_values_path: Path to the .npy file containing loglike values.
        output_path: Path where the histogram figure will be saved.
            Default: "loglike_histogram.png"
        bins: Number of bins for the histogram (for finite values).
            Default: 50

    Returns:
        str: Path to the saved figure.
    """
    with open(loglike_values_path, "rb") as f:
        values_array = np.load(f)
        _save_loglike_histogram(values_array, output_path=output_path, bins=bins)


def _save_loglike_histogram(
    loglike_values, output_path="loglike_histogram.png", bins=50
):
    """Generate and save a histogram of loglike values.

    Creates a histogram showing the distribution of loglike return values
    collected during pipeline execution. Separates successful values from
    -inf (error) values.

    Args:
        loglike_values: numpy array of loglike values collected during pipeline execution.
        output_path: Path where the histogram figure will be saved.
            Default: "loglike_histogram.png"
        bins: Number of bins for the histogram (for finite values).
            Default: 50

    Returns:
        str: Path to the saved figure.

    Raises:
        ValueError: If no loglike values have been collected yet.
    """

    # Separate finite values from -inf values

    finite_values = loglike_values[np.isfinite(loglike_values)]
    error_count = np.sum(~np.isfinite(loglike_values))

    # Create figure with histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(finite_values) > 0:
        ax.hist(
            finite_values,
            bins=bins,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="Finite values",
        )

    # Add text annotation for errors
    if error_count > 0:
        ax.text(
            0.02,
            0.98,
            f"Failed evaluations (-inf): {error_count}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Labels and title
    ax.set_xlabel("Log-Likelihood Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of Log-Likelihood Values", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Add statistics text
    if len(finite_values) > 0:
        stats_text = (
            f"Statistics (finite values only):\n"
            f"  Mean: {finite_values.mean():.4f}\n"
            f"  Std:  {finite_values.std():.4f}\n"
            f"  Min:  {finite_values.min():.4f}\n"
            f"  Max:  {finite_values.max():.4f}\n"
            f"  Count: {len(finite_values)}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            family="monospace",
        )

    plt.rcParams["text.usetex"] = False
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nHistogram saved to: {output_path}")
    if len(finite_values) > 0:
        print(f"  Finite values: {len(finite_values)}")
    if error_count > 0:
        print(f"  Error values (-inf): {error_count}")

    return output_path


def main():
    """Entry point for the clgpry-plot command-line tool."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: clgpry-plot <loglike_values.npy> [output_path]")
        sys.exit(1)

    loglike_values_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "loglike_histogram.png"

    save_loglike_histogram(loglike_values_path, output_path=output_path)


if __name__ == "__main__":
    main()
