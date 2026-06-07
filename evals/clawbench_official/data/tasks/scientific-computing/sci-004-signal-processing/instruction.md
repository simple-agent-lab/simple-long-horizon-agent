# FFT Signal Analysis and Filtering

## Task Description

You are provided with a noisy signal in the file `workspace/signal.csv`. This file contains two columns:

- `time`: The time points of the signal measurement (in seconds).
- `amplitude`: The amplitude of the signal at each time point.

Your task is to:

1. **Read** the signal data from `workspace/signal.csv`.
2. **Apply the Fast Fourier Transform (FFT)** to convert the time-domain signal into the frequency domain.
3. **Identify the dominant frequencies** present in the signal.
4. **Design a bandpass filter** that keeps frequencies between 5 Hz and 50 Hz, removing frequencies outside this range.
5. **Apply the bandpass filter** to the frequency-domain data.
6. **Reconstruct the filtered signal** back into the time domain using the inverse FFT.
7. **Write the following output files**:

   - `workspace/spectrum.csv`: Contains two columns:
     - `frequency` (Hz)
     - `magnitude` (absolute value of FFT coefficients)

   - `workspace/filtered_signal.csv`: Contains two columns:
     - `time` (seconds)
     - `filtered_amplitude` (filtered signal amplitude at each time point)

   - `workspace/signal_analysis.json`: A JSON file containing:
     - `dominant_frequencies`: List of the top 3 dominant frequencies (in Hz) sorted by magnitude descending.
     - `sampling_rate`: The sampling rate of the signal (Hz).
     - `num_samples`: Number of samples in the signal.

## Input File Format

- `workspace/signal.csv` (CSV)

| time (s) | amplitude |
|----------|------------|
| 0.000    | 0.123      |
| 0.001    | 0.456      |
| ...      | ...        |

## Output Files

- `workspace/spectrum.csv` (CSV)

| frequency (Hz) | magnitude |
|----------------|-----------|
| 0.0            | 0.0       |
| 1.0            | 5.6       |
| ...            | ...       |

- `workspace/filtered_signal.csv` (CSV)

| time (s) | filtered_amplitude |
|----------|--------------------|
| 0.000    | 0.12               |
| 0.001    | 0.45               |
| ...      | ...                |

- `workspace/signal_analysis.json` (JSON)

```json
{
  "dominant_frequencies": [12.0, 25.0, 7.0],
  "sampling_rate": 1000,
  "num_samples": 2048
}
```

## Notes

- Use the FFT to analyze the frequency components.
- The sampling rate can be inferred from the time column.
- The bandpass filter should zero out FFT coefficients outside the 5-50 Hz range.
- Use the inverse FFT to reconstruct the filtered signal.
- Ensure the output files are saved exactly as specified.

## Evaluation

Your solution will be tested on the correctness of the FFT analysis, filtering, and output files.

---