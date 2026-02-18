import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

t = np.linspace(0, 1, 1000)
dt = t[1] - t[0]
freq_hz = 5
clean = np.sin(2 * np.pi * freq_hz * t)

signal_power = np.mean(clean**2)
print("Signal Power:", signal_power)

noise_levels = [0.1, 0.5, 1.0]

plt.figure(figsize=(12, 8))

for noise_std in noise_levels:
    noisy = clean + np.random.normal(0, noise_std, size=t.shape)

    noise_component = noisy - clean
    noise_power = np.mean(noise_component**2)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"Noise std {noise_std} -> SNR (dB): {snr:.2f}")

    fft_vals = np.fft.fft(noisy)
    freqs = np.fft.fftfreq(len(t), dt)

    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])

    plt.plot(freqs_pos, magnitude, label=f"Noise std = {noise_std}")

plt.title("Frequency Spectrum with Increasing Noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()

plt.savefig("assets/frequency_spectrum.png", dpi=200)
plt.show()
