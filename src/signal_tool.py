import argparse
import os
import json
import wave
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

__version__ = "0.2.0"


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_supported_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".csv", ".wav"]


def list_supported_files(folder: str):
    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and is_supported_file(p):
            files.append(p)
    return sorted(files)


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


def process_one(input_path: str, fs_default: int, noise_std: float, seed, filt: str, cutoff: float,
                outdir: str, save_filtered: bool, no_plots: bool):
    ext = os.path.splitext(input_path)[1].lower()
    fs = fs_default

    if ext == ".csv":
        x = load_csv(input_path)
        label = os.path.basename(input_path)
    elif ext == ".wav":
        x, fs = load_wav_mono(input_path)
        label = os.path.basename(input_path)
    else:
        raise ValueError("Unsupported input. Use .csv or .wav.")

    if noise_std > 0.0:
        x_noisy = add_gaussian_noise(x, noise_std, seed)
    else:
        x_noisy = x

    if filt == "none":
        x_filtered = x_noisy
    else:
        x_filtered = lowpass_fft(x_noisy, fs, cutoff, mode=filt)

    outputs = {}

    base = os.path.splitext(os.path.basename(input_path))[0]
    file_outdir = os.path.join(outdir, base)
    ensure_dir(file_outdir)

    if save_filtered:
        csv_out = os.path.join(file_outdir, f"{base}_filtered.csv")
        np.savetxt(csv_out, x_filtered, delimiter=",")
        outputs["filtered_csv"] = csv_out.replace("\\", "/")

    if not no_plots:
        # time plot
        plt.figure(figsize=(12, 6))
        plt.plot(x_noisy, alpha=0.35, label="input/noisy")
        if filt != "none":
            plt.plot(x_filtered, linewidth=2, label=f"filtered ({filt}, cutoff={cutoff}Hz)")
        plt.title(f"Time Domain ({label})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        tpath = os.path.join(file_outdir, "time_domain.png")
        plt.savefig(tpath, dpi=200)
        plt.close()
        outputs["time_plot"] = tpath.replace("\\", "/")

        # fft plot
        f1, mag1 = compute_fft(x_noisy, fs)
        plt.figure(figsize=(12, 6))
        plt.plot(f1, mag1, label="input/noisy FFT")
        if filt != "none":
            f2, mag2 = compute_fft(x_filtered, fs)
            plt.plot(f2, mag2, label="filtered FFT")
        plt.xlim(0, min(200, fs / 2))
        plt.title(f"Frequency Domain (FFT) ({label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        fpath = os.path.join(file_outdir, "frequency_domain.png")
        plt.savefig(fpath, dpi=200)
        plt.close()
        outputs["fft_plot"] = fpath.replace("\\", "/")

    result = {
        "input": input_path.replace("\\", "/"),
        "fs_hz": fs,
        "n_samples": int(len(x)),
        "outputs": outputs
    }
    return result


def main():
    p = argparse.ArgumentParser(
        prog="signal_tool",
        description="Signal processing tool: load or simulate signals, add noise, FFT, low-pass filter, save outputs."
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    p.add_argument("--input", type=str, default=None,
                   help="Path to input file (.csv or .wav) OR a folder containing .csv/.wav files. If omitted, simulates a sine wave.")
    p.add_argument("--fs", type=int, default=1000, help="Sampling rate (Hz). Used for simulation or CSV data.")
    p.add_argument("--duration", type=float, default=1.0, help="Duration (s) for simulation.")
    p.add_argument("--freq", type=float, default=5.0, help="Sine frequency (Hz) for simulation.")

    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std to add (0 = none).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (use -1 for no seeding).")

    p.add_argument("--filter", choices=["none", "hard", "gaussian"], default="none",
                   help="Low-pass filter type in frequency domain.")
    p.add_argument("--cutoff", type=float, default=10.0, help="Low-pass cutoff frequency (Hz).")

    p.add_argument("--outdir", type=str, default="assets", help="Directory to save outputs.")
    p.add_argument("--batch-save-filtered", action="store_true",
                   help="(Batch mode) Save filtered CSV for each file.")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting/saving plots.")
    p.add_argument("--report", type=str, default=None, help="Save a JSON report to this path.")

    args = p.parse_args()

    if args.version:
        print(__version__)
        return

    seed = None if args.seed == -1 else args.seed
    ensure_dir(args.outdir)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": __version__,
        "args": vars(args),
        "mode": None,
        "results": []
    }

    # Mode selection
    if args.input is None:
        # single simulate mode (no batch)
        x = generate_sine(args.duration, args.fs, args.freq)
        if args.noise_std > 0.0:
            x_noisy = add_gaussian_noise(x, args.noise_std, seed)
        else:
            x_noisy = x

        if args.filter == "none":
            x_filtered = x_noisy
        else:
            x_filtered = lowpass_fft(x_noisy, args.fs, args.cutoff, mode=args.filter)

        # save plots at root outdir
        if not args.no_plots:
            plt.figure(figsize=(12, 6))
            plt.plot(x_noisy, alpha=0.35, label="input/noisy")
            if args.filter != "none":
                plt.plot(x_filtered, linewidth=2, label=f"filtered ({args.filter}, cutoff={args.cutoff}Hz)")
            plt.plot(x, linestyle="--", label="clean (ground truth)")
            plt.title("Time Domain (simulated)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            tpath = os.path.join(args.outdir, "time_domain.png")
            plt.savefig(tpath, dpi=200)
            plt.close()

            f1, mag1 = compute_fft(x_noisy, args.fs)
            plt.figure(figsize=(12, 6))
            plt.plot(f1, mag1, label="input/noisy FFT")
            if args.filter != "none":
                f2, mag2 = compute_fft(x_filtered, args.fs)
                plt.plot(f2, mag2, label="filtered FFT")
            plt.xlim(0, min(200, args.fs / 2))
            plt.title("Frequency Domain (FFT) (simulated)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.legend()
            plt.tight_layout()
            fpath = os.path.join(args.outdir, "frequency_domain.png")
            plt.savefig(fpath, dpi=200)
            plt.close()

        report["mode"] = "simulate"
        report["results"].append({
            "input": "simulated",
            "fs_hz": args.fs,
            "n_samples": int(len(x)),
            "outputs": {
                "time_plot": os.path.join(args.outdir, "time_domain.png").replace("\\", "/") if not args.no_plots else None,
                "fft_plot": os.path.join(args.outdir, "frequency_domain.png").replace("\\", "/") if not args.no_plots else None
            }
        })
    else:
        # file or folder mode
        if os.path.isdir(args.input):
            report["mode"] = "batch"
            files = list_supported_files(args.input)
            if len(files) == 0:
                raise ValueError("No .csv or .wav files found in the folder.")

            for fp in files:
                res = process_one(
                    input_path=fp,
                    fs_default=args.fs,
                    noise_std=args.noise_std,
                    seed=seed,
                    filt=args.filter,
                    cutoff=args.cutoff,
                    outdir=args.outdir,
                    save_filtered=args.batch_save_filtered,
                    no_plots=args.no_plots
                )
                report["results"].append(res)
                print(f"Processed: {os.path.basename(fp)}")
        else:
            report["mode"] = "single_file"
            if not is_supported_file(args.input):
                raise ValueError("Unsupported input. Use .csv or .wav.")
            res = process_one(
                input_path=args.input,
                fs_default=args.fs,
                noise_std=args.noise_std,
                seed=seed,
                filt=args.filter,
                cutoff=args.cutoff,
                outdir=args.outdir,
                save_filtered=False,
                no_plots=args.no_plots
            )
            report["results"].append(res)

    if args.report is not None:
        ensure_dir(os.path.dirname(args.report) or ".")
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
