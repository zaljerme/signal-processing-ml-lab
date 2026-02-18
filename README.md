# Signal Processing + ML Lab

A small, reproducible signal-processing CLI tool that can simulate or load signals, add noise, compute FFT spectra, apply frequency-domain low-pass filtering, and export plots + JSON reports.  
Built to be clean, scriptable, and easy to extend into ML denoising/classification projects.

## Features
- Simulate sine waves or load `.csv` / `.wav`
- Add Gaussian noise (seeded for reproducibility)
- FFT visualization (frequency spectrum)
- Frequency-domain low-pass filtering (hard cutoff or gaussian)
- Batch mode: process an entire folder of files
- Exports plots and a structured JSON report

## Install
```bash
pip install -r requirements.txt
python -m src --version
python -m src --freq 5 --fs 1000 --duration 1 --noise-std 0.8 --filter gaussian --cutoff 10 --report assets/report.json
python -m src --input data_folder --noise-std 0.2 --filter gaussian --cutoff 15 --batch-save-filtered --report assets/demo/batch_report.json
