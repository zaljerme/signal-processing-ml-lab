# Signal Processing + ML Lab

## Signal Generation and Noise Experiment

### Objective
Analyze how increasing Gaussian noise affects a 5 Hz sine wave.

### Method
- Generated a clean sine wave
- Added Gaussian noise with std values: 0.1, 0.5, 1.0
- Computed Signal to Noise Ratio (SNR) in decibels
- Visualized results and saved output image

### Results
SNR decreases as noise increases:
- 0.1 -> High SNR
- 0.5 -> Moderate SNR
- 1.0 -> Negative SNR (noise dominates signal)

### Output
See assets/noise_comparison.png


## Frequency Domain (FFT)

### Objective
Visualize how Gaussian noise affects the frequency spectrum of the sine wave.

### What to look for
A strong peak remains near 5 Hz, while the noise raises the overall spectrum baseline as noise increases.

### Output
See assets/frequency_spectrum.png


## Low-Pass Filtering

### Objective
Reduce broadband noise while preserving the 5 Hz sine wave.

### Method
Applied a frequency-domain low-pass filter (cutoff = 10 Hz) by masking FFT bins and using inverse FFT.

### Output
See assets/filtered_signal.png


## CLI Tool: signal_tool.py

### Simulate + add noise + FFT + filter (saves plots to assets/)
python .\src\signal_tool.py --freq 5 --fs 1000 --duration 1 --noise-std 0.8 --filter gaussian --cutoff 10

### Load CSV (single column), filter, save filtered CSV
python .\src\signal_tool.py --input your_data.csv --fs 1000 --filter hard --cutoff 10 --save-filtered filtered.csv

### Load WAV, filter, save filtered WAV
python .\src\signal_tool.py --input your_audio.wav --filter gaussian --cutoff 2000 --save-filtered filtered.wav


### Example with JSON report
python -m src --freq 5 --fs 1000 --duration 1 --noise-std 0.8 --filter gaussian --cutoff 10 --report assets/report.json

