import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

# Suppress runtime warnings for cleaner output during optimization
warnings.filterwarnings("ignore")


def von_mises_discrete(x, mu, kappa):
    """
    Computes a normalized, discrete von Mises distribution over 360 degrees.
    Designed to prevent numerical instability at high kappa values by standardizing
    before the exponential calculation.
    """
    x_rad = np.radians(x)
    mu_rad = np.radians(mu)
    pdf = np.exp(kappa * np.cos(x_rad - mu_rad))
    return pdf / np.sum(pdf)


def nll_switching_observer(params, df_subject):
    """
    Calculates the Negative Log-Likelihood of the Switching Observer model
    generating the subject's actual trial estimates.
    """
    kl24, kl12, kl6, kp80, kp40, kp20, kp10, prand, km = params

    # Boundary constraints to prevent invalid probabilities or negative concentrations
    if any(k <= 0 for k in [kl24, kl12, kl6, kp80, kp40, kp20, kp10, km]) or not (
        0 <= prand <= 1
    ):
        return 1e9

    # Map experimental conditions to the respective fit parameters
    kl_map = {0.24: kl24, 0.12: kl12, 0.06: kl6}
    kp_map = {80: kp80, 40: kp40, 20: kp20, 10: kp10}

    # Discretized circular space from 1 to 360 degrees
    q = np.arange(1, 361)

    # Motor noise distribution (in frequency domain for fast convolution)
    v_motor = von_mises_discrete(q, 0, km)
    v_motor_fft = np.fft.fft(v_motor)

    total_nll = 0.0

    # Group trials by identical conditions to compute distributions efficiently
    for (coh, prior_std, q_true), group in df_subject.groupby(
        ["motion_coherence", "prior_std", "motion_direction"]
    ):
        ke = kl_map.get(coh, kl24)
        kp = kp_map.get(prior_std, kp80)

        # Calculate switching probabilities
        p_prior = kp / (kp + ke)
        p_e = 1.0 - p_prior

        # Sensory evidence distribution
        v_e = von_mises_discrete(q, q_true, ke)

        # Delta function at the prior mean (225 degrees)
        delta_prior = np.zeros(360)
        delta_prior[225 % 360 - 1] = 1.0

        # Core Switching mechanism mixture distribution
        p_percept = (1 - prand) * (p_e * v_e + p_prior * delta_prior) + (prand / 360.0)

        # Apply Motor Noise via Circular Convolution
        p_est = np.real(np.fft.ifft(np.fft.fft(p_percept) * v_motor_fft))
        p_est = np.clip(p_est, 1e-10, 1.0)  # Prevent log(0) errors
        p_est /= np.sum(p_est)

        # Extract the subject's actual estimates in degrees (1 to 360)
        est_rad = np.arctan2(group["estimate_y"].values, group["estimate_x"].values)
        est_deg = np.degrees(est_rad) % 360
        est_deg[est_deg == 0] = 360
        est_idx = np.round(est_deg).astype(int) - 1

        # Accumulate Log-Likelihood
        probs = p_est[est_idx]
        total_nll -= np.sum(np.log(probs))

    return total_nll


def extract_table_s3b_parameters(csv_filepath="data.csv"):
    """
    Iterates through all subjects in the dataset, fits the Switching Observer
    model to their behavioral estimates, and returns a DataFrame mimicking Table S3b.
    """
    df = pd.read_csv(csv_filepath)
    df = df.dropna(
        subset=[
            "estimate_x",
            "estimate_y",
            "motion_direction",
            "motion_coherence",
            "prior_std",
        ]
    )

    results = []
    subject_ids = sorted(df["subject_id"].unique())

    # A standard initial guess (can be adjusted or randomized to avoid local minima)
    init_guess = [50.0, 10.0, 2.0, 0.5, 1.0, 5.0, 20.0, 0.05, 15.0]

    for subj in subject_ids:
        print(f"Optimizing parameters for Subject {subj}...")
        df_sub = df[df["subject_id"] == subj]

        # Run the Nelder-Mead optimization
        res = minimize(
            nll_switching_observer,
            init_guess,
            args=(df_sub,),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-4, "fatol": 1e-4},
        )

        # Store results
        if res.success:
            params = res.x
            results.append(
                {
                    "Subject": subj,
                    "Kl24": round(params[0], 2),
                    "Kl12": round(params[1], 2),
                    "Kl6": round(params[2], 2),
                    "Kp80": round(params[3], 2),
                    "Kp40": round(params[4], 2),
                    "Kp20": round(params[5], 2),
                    "Kp10": round(params[6], 2),
                    "Prand": round(params[7], 3),
                    "Km": round(params[8], 2),
                }
            )
            print(results[-1])
        else:
            res.message = res.message if hasattr(res, "message") else "No message"
            print(f"Optimization failed for Subject {subj}: {res.message}")

    return pd.DataFrame(results).set_index("Subject")


if __name__ == "__main__":
    table_s3b = extract_table_s3b_parameters("files/data.csv")
    print(table_s3b)
