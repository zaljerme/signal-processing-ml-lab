import argparse
import os
import json
import wave
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

__version__ = "0.3.0"


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


# ---------------- FEATURE EXTRACTION ----------------

def extract_features(x: np.ndarray, fs: int) -> dict:
    rms = float(np.sqrt(np.mean(x ** 2)))
    mean = float(np.mean(x))
    std = float(np.std(x))

    freqs, mag = compute_fft(x, fs)
    peak_freq = float(freqs[np.argmax(mag)])
    spectral_centroid = float(np.sum(freqs * mag) / np.sum(mag))

    return {
        "rms": rms,
        "mean": mean,
        "std": std,
        "peak_frequency_hz": peak_freq,
        "spectral_centroid_hz": spectral_centroid
    }


# ---------------- PROCESS ONE FILE ----------------

def process_one(input_path: str, fs_default: int, noise_std: float, seed,
                filt: str, cutoff: float, outdir: str,
                save_filtered: bool, no_plots: bool):

    ext = os.path.splitext(input_path)[1].lower()
    fs = fs_default

    if ext == ".csv":
        x = load_csv(input_path)
    elif ext == ".wav":
        x, fs = load_wav_mono(input_path)
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

    result = {
        "input": input_path.replace("\\", "/"),
        "fs_hz": fs,
        "n_samples": int(len(x)),
        "features_noisy": extract_features(x_noisy, fs),
        "features_filtered": extract_features(x_filtered, fs),
        "outputs": outputs
    }

    return result


# ---------------- MAIN ----------------

def main():
    p = argparse.ArgumentParser(
        prog="signal_tool",
        description="Signal processing CLI tool."
    )

    p.add_argument("--version", action="store_true")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--fs", type=int, default=1000)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--freq", type=float, default=5.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter", choices=["none", "hard", "gaussian"], default="none")
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--outdir", type=str, default="assets")
    p.add_argument("--batch-save-filtered", action="store_true")
    p.add_argument("--report", type=str, default=None)

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
        "results": []
    }

    if args.input is None:
        x = generate_sine(args.duration, args.fs, args.freq)
        if args.noise_std > 0:
            x = add_gaussian_noise(x, args.noise_std, seed)

        if args.filter != "none":
            x = lowpass_fft(x, args.fs, args.cutoff, args.filter)

        report["results"].append({
            "input": "simulated",
            "fs_hz": args.fs,
            "n_samples": len(x),
            "features": extract_features(x, args.fs)
        })

    elif os.path.isdir(args.input):
        files = list_supported_files(args.input)
        for fp in files:
            res = process_one(
                fp, args.fs, args.noise_std, seed,
                args.filter, args.cutoff,
                args.outdir, args.batch_save_filtered, False
            )
            report["results"].append(res)

    else:
        res = process_one(
            args.input, args.fs, args.noise_std, seed,
            args.filter, args.cutoff,
            args.outdir, False, False
        )
        report["results"].append(res)

    if args.report:
        ensure_dir(os.path.dirname(args.report) or ".")
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {args.report}")


if __name__ == "__main__":
    main()
