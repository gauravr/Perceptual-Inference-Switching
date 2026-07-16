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
    Switching Bayesian observer formulation[cite: 1].
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

        # Competitive switching probabilities[cite: 1]
        p_prior = kp / (kp + ke)
        p_e = 1 - p_prior

        # Percept probability distribution before motor noise
        percept = p_e * von_mises(theta_rad, rel_motion_rad, ke)
        delta = np.zeros_like(theta_rad)
        delta[180] = 1.0 / np.radians(1.0)  # Delta strictly at prior mean
        percept += p_prior * delta

        # Motor noise distribution[cite: 1]
        motor_noise = von_mises(theta_rad, 0, km)

        # Convert densities to probability masses for FFT circular convolution[cite: 1]
        percept_pdf = percept * np.radians(1.0)
        motor_pdf = motor_noise * np.radians(1.0)

        convolved = np.real(
            np.fft.ifft(np.fft.fft(percept_pdf) * np.fft.fft(motor_pdf))
        )
        convolved = np.roll(convolved, 180)  # Shift zero frequency to center
        convolved_density = convolved / np.radians(1.0)

        # Introduce uniform lapse rate (Prand)[cite: 1]
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
        label="Switching Bayesian Observer Model Prediction",
    )
