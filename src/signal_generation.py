import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Time setup
t = np.linspace(0, 1, 1000)
dt = t[1] - t[0]
freq_hz = 5
clean = np.sin(2 * np.pi * freq_hz * t)

signal_power = np.mean(clean**2)
print("Signal Power:", signal_power)

noise_std = 0.8  # stronger noise for demonstration
noisy = clean + np.random.normal(0, noise_std, size=t.shape)

# Compute SNR before filtering
noise_component = noisy - clean
noise_power = np.mean(noise_component**2)
snr_before = 10 * np.log10(signal_power / noise_power)
print(f"SNR before filtering: {snr_before:.2f} dB")

# FFT
fft_vals = np.fft.fft(noisy)
freqs = np.fft.fftfreq(len(t), dt)

# Low-pass filter (keep frequencies under 10 Hz)
cutoff = 10
filter_mask = np.abs(freqs) < cutoff
fft_filtered = fft_vals * filter_mask

# Inverse FFT
filtered = np.fft.ifft(fft_filtered).real

# Compute SNR after filtering
noise_component_after = filtered - clean
noise_power_after = np.mean(noise_component_after**2)
snr_after = 10 * np.log10(signal_power / noise_power_after)
print(f"SNR after filtering: {snr_after:.2f} dB")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(t, noisy, alpha=0.4, label="Noisy Signal")
plt.plot(t, filtered, linewidth=2, label="Filtered Signal")
plt.plot(t, clean, linestyle="--", label="Clean Signal")
plt.legend()
plt.title("Low-Pass Filtering of Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()

plt.savefig("assets/filtered_signal.png", dpi=200)
plt.show()
