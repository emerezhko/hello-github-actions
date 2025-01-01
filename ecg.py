import numpy as np
import matplotlib.pyplot as plt

def generate_realistic_ecg(length=1000, sampling_rate=500, bpm=60, noise_level=0.01, message=None):
    """
    Generate a realistic ECG waveform with optional steganography encoding.
    
    Parameters:
    - length: int, total length of the signal (number of points).
    - sampling_rate: int, number of samples per second.
    - bpm: int, beats per minute to determine heart rate.
    - noise_level: float, standard deviation of Gaussian noise added to the signal.
    - message: str, optional string to encode into the waveform.
    
    Returns:
    - time: np.ndarray, time axis for the signal.
    - ecg_wave: np.ndarray, generated ECG-like signal.
    """
    # Time axis
    time = np.linspace(0, length / sampling_rate, length)
    
    # Define heart rate
    heart_rate = bpm / 60  # Beats per second
    period = 1 / heart_rate  # Duration of one heartbeat
    num_beats = int(length / sampling_rate * heart_rate)
    
    # ECG waveform components
    def p_wave(t):
        return 0.1 * np.sin(np.pi * t / 0.2) * (t >= 0) * (t <= 0.2)
    
    def q_wave(t):
        return -0.15 * np.exp(-((t - 0.1) / 0.03)**2)
    
    def r_wave(t):
        return 1.0 * np.exp(-((t - 0.15) / 0.02)**2)
    
    def s_wave(t):
        return -0.2 * np.exp(-((t - 0.2) / 0.03)**2)
    
    def t_wave(t):
        return 0.35 * np.sin(np.pi * (t - 0.3) / 0.2) * (t >= 0.3) * (t <= 0.5)
    
    # Generate waveform for one heartbeat
    def single_heartbeat(t):
        p = p_wave(t)
        q = q_wave(t)
        r = r_wave(t)
        s = s_wave(t)
        t = t_wave(t)
        return p + q + r + s + t

    # Build complete ECG signal
    ecg_wave = np.zeros_like(time)
    for i in range(num_beats):
        start_idx = int(i * period * sampling_rate)
        end_idx = int((i + 1) * period * sampling_rate)
        t_segment = np.linspace(0, period, end_idx - start_idx)
        ecg_wave[start_idx:end_idx] += single_heartbeat(t_segment)

    # Add Gaussian noise
    ecg_wave += np.random.normal(0, noise_level, size=length)

    # Encode message via steganography
    if message:
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        step = max(1, length // len(binary_message))
        for i, bit in enumerate(binary_message):
            if bit == '1':
                ecg_wave[i * step] += 0.02  # Add a small "bump" for encoding

    return time, ecg_wave


# Parameters
params = {
    "length": 2000,
    "sampling_rate": 500,
    "bpm": 75,
    "noise_level": 0.02,
    "message": "HELLO"
}

# Generate and plot the ECG waveform
time, ecg_wave = generate_realistic_ecg(**params)

plt.figure(figsize=(12, 4))
plt.plot(time, ecg_wave, label='Realistic ECG Signal')
plt.title("Realistic ECG Waveform with Steganography")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
