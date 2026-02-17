import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # reproducible noise

t = np.linspace(0, 1, 1000)
freq_hz = 5
clean = np.sin(2 * np.pi * freq_hz * t)

signal_power = np.mean(clean**2)
print("Signal Power:", signal_power)

noise_levels = [0.1, 0.5, 1.0]

plt.figure(figsize=(10, 6))

for noise_std in noise_levels:
    noisy = clean + np.random.normal(0, noise_std, size=t.shape)

    noise_component = noisy - clean
    noise_power = np.mean(noise_component**2)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"Noise std {noise_std} -> SNR (dB): {snr:.2f}")

    plt.plot(t, noisy, label=f"Noise std = {noise_std}")

plt.plot(t, clean, linewidth=2, color="black", label="Clean Signal")
plt.title("Effect of Increasing Gaussian Noise on a Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()

plt.savefig("assets/noise_comparison.png", dpi=200)
plt.show()
