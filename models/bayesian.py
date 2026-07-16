import numpy as np
from scipy.special import i0
import warnings

from plotting import plot_histograms

# Suppress warnings from FFT and empty data slices
warnings.filterwarnings("ignore")


def von_mises(theta, mu, kappa):
    """Calculates the von Mises probability density."""
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))


def compute(theta_deg, rel_motion_deg, prior_std, params, coherence=0.06):
    """
    Computes the bimodal percept probability distribution based on the
    Basic Bayesian observer formulation[cite: 1].

    Args:
        theta_deg (ndarray): The evaluation angles in degrees.
        rel_motion_deg (float): The actual motion direction relative to the prior mean.
        prior_std (int): The standard deviation of the prior (e.g., 80, 40, 20, 10).
        params (DataFrame): The parameter dataset containing fitted strengths.
        coherence (float): The motion coherence level.

    Returns:
        ndarray: The averaged predicted probability density function over all subjects.
    """
    theta_rad = np.radians(theta_deg)
    rel_motion_rad = np.radians(rel_motion_deg)
    preds = []

    # Map coherence to the correct likelihood parameter column
    kl_col = "Kl6" if coherence == 0.06 else ("Kl12" if coherence == 0.12 else "Kl24")

    # Average predictions across all subject parameters
    for _, row in params.iterrows():
        ke = row[kl_col]
        kp = row[f"Kp{prior_std}"]
        km = row["Km"]
        pr = row["Prand"]

        # 1. Simulate sensory evidence distribution P(theta_e | theta_true)[cite: 1]
        # Represents trial-to-trial variability of sensory evidence
        prob_e = von_mises(theta_rad, rel_motion_rad, ke) * np.radians(1.0)

        # 2. Compute posterior mode (MAP estimate) theta_p for each theta_e[cite: 1]
        # The posterior is the product of likelihood V(theta; theta_e, ke) and prior V(theta; 0, kp)[cite: 1]
        # Using trigonometric addition, the mode of this product is:
        theta_p = np.arctan2(ke * np.sin(theta_rad), kp + ke * np.cos(theta_rad))

        # 3. Build the percept distribution P(theta_p | theta_true)[cite: 1]
        percept_mass = np.zeros_like(theta_rad)
        deg_p = np.degrees(theta_p)

        # Map degrees to array indices (shifted by +180 to handle negative angles)
        idx = np.round(deg_p).astype(int) + 180
        idx = np.clip(idx, 0, 360)

        # Accumulate probability mass into the deterministic MAP estimate bins
        for i in range(len(prob_e)):
            percept_mass[idx[i]] += prob_e[i]

        # Handle circular boundary conditions (-180 and +180 are the same)
        percept_mass[0] += percept_mass[360]
        percept_mass[360] = percept_mass[0]
        percept_mass = percept_mass / np.sum(percept_mass)  # Normalize

        # 4. Convolve percept distribution with Motor Noise V(theta; 0, km)[cite: 1]
        motor_noise = von_mises(theta_rad, 0, km)
        motor_pdf = motor_noise * np.radians(1.0)

        # Apply circular convolution via FFT
        convolved = np.real(
            np.fft.ifft(np.fft.fft(percept_mass) * np.fft.fft(motor_pdf))
        )
        convolved = np.roll(convolved, 180)  # Shift zero frequency to center
        convolved_density = convolved / np.radians(1.0)

        # 5. Introduce uniform lapse rate (Prand)[cite: 1]
        final_pdf = (1 - pr) * convolved_density + pr / (2 * np.pi)
        final_pdf_deg = final_pdf * np.pi / 180.0
        preds.append(final_pdf_deg)

    return np.mean(preds, axis=0)


# Execute the main plotting function
if __name__ == "__main__":
    plot_histograms(
        "files/data.csv",
        "files/params.csv",
        compute,
        label="Bayesian Observer Model Prediction",
    )
