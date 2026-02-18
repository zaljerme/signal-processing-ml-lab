import argparse
import os
import json
import wave
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Version (must be ABOVE main so --version works)
__version__ = "0.1.0"


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim == 2:
        data = data[:, 0]
    data = data[~np.isnan(data)]
    return data.astype(float)


def load_wav_mono(path: str):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels)[:, 0]

    return x, fr


def save_wav_mono(path: str, x: np.ndarray, fr: int) -> None:
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fr)
        wf.writeframes(pcm)


def generate_sine(duration: float, fs: int, freq_hz: float) -> np.ndarray:
    t = np.arange(int(duration * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def add_gaussian_noise(x: np.ndarray, noise_std: float, seed):
    if seed is not None:
        np.random.seed(seed)
    return x + np.random.normal(0.0, noise_std, size=x.shape)


def compute_fft(x: np.ndarray, fs: int):
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.fft(x)
    pos = freqs >= 0
    return freqs[pos], np.abs(fft_vals[pos])


def lowpass_fft(x: np.ndarray, fs: int, cutoff_hz: float, mode: str = "hard") -> np.ndarray:
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    X = np.fft.fft(x)

    if mode == "hard":
        mask = (np.abs(freqs) <= cutoff_hz).astype(float)
    elif mode == "gaussian":
        sigma = cutoff_hz / 2.0 if cutoff_hz > 0 else 1.0
        mask = np.exp(-(freqs ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError("mode must be 'hard' or 'gaussian'")

    Y = X * mask
    return np.fft.ifft(Y).real


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((test - clean) ** 2)
    return 10.0 * np.log10(signal_power / noise_power)


def main():
    p = argparse.ArgumentParser(
        prog="signal_tool",
        description="Signal processing tool: load or simulate signals, add noise, FFT, low-pass filter, save outputs."
    )

    # Professional nicety
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    # Input / simulation
    p.add_argument("--input", type=str, default=None,
                   help="Path to input file (.csv or .wav). If omitted, simulates a sine wave.")
    p.add_argument("--fs", type=int, default=1000,
                   help="Sampling rate (Hz). Used for simulation or CSV data.")
    p.add_argument("--duration", type=float, default=1.0,
                   help="Duration (s) for simulation.")
    p.add_argument("--freq", type=float, default=5.0,
                   help="Sine frequency (Hz) for simulation.")

    # Noise
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Gaussian noise std to add (0 = none).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility (use -1 for no seeding).")

    # Filtering
    p.add_argument("--filter", choices=["none", "hard", "gaussian"], default="none",
                   help="Low-pass filter type in frequency domain.")
    p.add_argument("--cutoff", type=float, default=10.0,
                   help="Low-pass cutoff frequency (Hz).")

    # Outputs
    p.add_argument("--outdir", type=str, default="assets",
                   help="Directory to save plots.")
    p.add_argument("--save-filtered", type=str, default=None,
                   help="Save filtered signal to this path (.csv or .wav).")
    p.add_argument("--no-plots", action="store_true",
                   help="Disable plotting/saving plots.")
    p.add_argument("--report", type=str, default=None,
                   help="Save a JSON report to this path (e.g., assets/report.json).")

    args = p.parse_args()

    # Version exit
    if args.version:
        print(__version__)
        return

    seed = None if args.seed == -1 else args.seed
    ensure_dir(args.outdir)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "fs_hz": None,
        "n_samples": None,
        "metrics": {},
        "outputs": {}
    }

    clean = None
    fs = args.fs

    # Load or simulate
    if args.input is None:
        clean = generate_sine(args.duration, fs, args.freq)
        x = clean.copy()
        source_label = f"simulated_sine_{args.freq}Hz"
    else:
        ext = os.path.splitext(args.input)[1].lower()
        if ext == ".csv":
            x = load_csv(args.input)
            source_label = os.path.basename(args.input)
        elif ext == ".wav":
            x, fs = load_wav_mono(args.input)
            source_label = os.path.basename(args.input)
        else:
            raise ValueError("Unsupported input. Use .csv or .wav.")

    report["fs_hz"] = fs
    report["n_samples"] = int(len(x))

    # Noise
    if args.noise_std > 0.0:
        x_noisy = add_gaussian_noise(x, args.noise_std, seed)
    else:
        x_noisy = x

    # Filter
    if args.filter == "none":
        x_filtered = x_noisy
    else:
        x_filtered = lowpass_fft(x_noisy, fs, args.cutoff, mode=args.filter)

    # Metrics (SNR only if we have clean ground truth)
    if clean is not None and args.noise_std > 0.0:
        report["metrics"]["snr_noisy_db"] = float(snr_db(clean, x_noisy))
        print(f"SNR before filtering (noisy): {report['metrics']['snr_noisy_db']:.2f} dB")

    if clean is not None and args.filter != "none":
        report["metrics"]["snr_filtered_db"] = float(snr_db(clean, x_filtered))
        print(f"SNR after filtering ({args.filter}): {report['metrics']['snr_filtered_db']:.2f} dB")

    # Save filtered output
    if args.save_filtered is not None:
        out_ext = os.path.splitext(args.save_filtered)[1].lower()
        ensure_dir(os.path.dirname(args.save_filtered) or ".")
        if out_ext == ".csv":
            np.savetxt(args.save_filtered, x_filtered, delimiter=",")
        elif out_ext == ".wav":
            save_wav_mono(args.save_filtered, x_filtered, fs)
        else:
            raise ValueError("save-filtered must end with .csv or .wav")
        report["outputs"]["filtered_path"] = args.save_filtered
        print(f"Saved filtered output -> {args.save_filtered}")

    # Plots
    if not args.no_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(x_noisy, alpha=0.35, label="input/noisy")
        if args.filter != "none":
            plt.plot(x_filtered, linewidth=2, label=f"filtered ({args.filter}, cutoff={args.cutoff}Hz)")
        if clean is not None:
            plt.plot(clean, linestyle="--", label="clean (ground truth)")
        plt.title(f"Time Domain ({source_label})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        time_path = os.path.join(args.outdir, "time_domain.png")
        plt.savefig(time_path, dpi=200)
        plt.close()
        report["outputs"]["time_plot"] = time_path.replace("\\", "/")
        print(f"Saved plot -> {time_path}")

        f1, mag1 = compute_fft(x_noisy, fs)
        plt.figure(figsize=(12, 6))
        plt.plot(f1, mag1, label="input/noisy FFT")
        if args.filter != "none":
            f2, mag2 = compute_fft(x_filtered, fs)
            plt.plot(f2, mag2, label="filtered FFT")
        plt.xlim(0, min(200, fs / 2))
        plt.title(f"Frequency Domain (FFT) ({source_label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        fft_path = os.path.join(args.outdir, "frequency_domain.png")
        plt.savefig(fft_path, dpi=200)
        plt.close()
        report["outputs"]["fft_plot"] = fft_path.replace("\\", "/")
        print(f"Saved plot -> {fft_path}")

    # Report
    if args.report is not None:
        ensure_dir(os.path.dirname(args.report) or ".")
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
