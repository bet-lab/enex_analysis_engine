import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enex_analysis.g_function import precompute_gfunction

def main():
    print("Testing refactored g_function interpolation (No Hybrid, Pure Geomspace + UHF)...")

    dt_s = 60.0
    t_max_s = 24 * 3600.0 # 24 hours
    
    # 1. Precompute using the newly modified code
    gfunc_interp = precompute_gfunction(
        N_1=1, N_2=1, B=6.0, H_b=150.0, D_b=2.0, r_b=0.075,
        alpha_s=1.0e-6, k_s=2.0, t_max_s=t_max_s, dt_s=dt_s
    )
    
    # 2. Simulate the array that GSHPB steps through over 24 hours
    # 1440 points (60s intervals)
    times_sim = np.arange(60.0, t_max_s + 60.0, 60.0)
    g_interpolated_vals = gfunc_interp(times_sim)
    
    # Check if there is any numerical "noise" / NaN / negative
    min_val = np.min(g_interpolated_vals)
    max_val = np.max(g_interpolated_vals)
    print(f"Interpolation completed over 24 hours.")
    print(f"Min G-value: {min_val:.5e}")
    print(f"Max G-value: {max_val:.5e}")
    if min_val < 0.0:
        print("WARNING: Negative values generated!")

    # 3. Plot the result to prove there's NO noise
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(times_sim, g_interpolated_vals, label=f"Interpolated G-function (dt={dt_s}s)", color="#3b82f6", linewidth=2)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("G-function [mK/W]")
    ax.set_title("Interpolation Output Verification (60s Timesteps over 24 hours)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    
    # Save the output
    fig.tight_layout()
    out_path = "/home/habin/Codes/enex_engine/enex_analysis_engine/temp/verify_logspace_interpolation.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
