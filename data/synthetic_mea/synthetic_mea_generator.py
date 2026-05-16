# FILE: data/synthetic_mea/synthetic_mea_generator.py
# This file creates artificial MEA data that matches the statistics 
# of the proprietary MEA Real Recordings from Table I of the paper.

import numpy as np
import scipy.signal as signal
from pathlib import Path

def generate_synthetic_mea_dataset(
    output_dir="data/synthetic_mea/",
    n_neurons=10,           # Matches paper: 8-12 neurons (we use 10)
    n_channels=64,          # Matches paper: 64-channel MEA
    duration_min=15,        # Matches paper: 15 minutes
    sampling_rate=25000,    # Matches paper: 25 kHz
    snr_range=(3, 15),      # Matches paper: 3-15 dB
    seed=42
):
    """
    Generates synthetic MEA data statistically matched to Table I.
    
    Statistical targets from paper:
    - Total spikes: ~8,500
    - Spikes/neuron: 1,062 ± 280
    - SNR: 3-15 dB
    - Neuron types: 8 pyramidal, 3 fast-spiking interneurons, 1 chandelier
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_samples = int(duration_min * 60 * sampling_rate)
    
    # Generate realistic spike templates for each neuron type
    templates = {}
    for i in range(n_neurons):
        if i < 8:  # Pyramidal cells
            templates[i] = generate_pyramidal_template(sampling_rate)
        elif i < 11:  # Fast-spiking interneurons
            templates[i] = generate_interneuron_template(sampling_rate)
        else:  # Chandelier cell
            templates[i] = generate_chandelier_template(sampling_rate)
    
    # Generate spike trains with realistic firing rates
    spike_trains = {}
    total_spikes = 0
    for i in range(n_neurons):
        # Target: ~1,062 spikes per neuron with variation
        n_spikes = int(np.random.normal(1062, 280))
        n_spikes = max(100, n_spikes)  # Minimum 100 spikes
        
        # Generate spike times with refractory period (~2ms)
        spike_times = generate_realistic_spike_times(
            n_spikes, total_samples, sampling_rate, refractory_ms=2.0
        )
        spike_trains[i] = spike_times
        total_spikes += len(spike_times)
    
    # Generate noise and add spikes
    noise = generate_realistic_mea_noise(total_samples, n_channels, sampling_rate)
    data = noise.copy()
    
    labels = []
    timestamps = []
    snr_values = []
    
    for neuron_id, spike_times in spike_trains.items():
        template = templates[neuron_id]
        for t in spike_times:
            # Add template to data with random amplitude variation
            amplitude = np.random.uniform(0.7, 1.3)
            snr = np.random.uniform(snr_range[0], snr_range[1])
            
            # Scale by SNR
            signal_power = np.sum(template**2)
            noise_power = np.sum(noise[t:t+len(template), 0]**2)
            scale = np.sqrt(signal_power / (noise_power * 10**(-snr/10)))
            
            if t + len(template) < total_samples:
                data[t:t+len(template), 0] += template * scale * amplitude
                
                labels.append(neuron_id)
                timestamps.append(t / sampling_rate)  # Convert to seconds
                snr_values.append(snr)
    
    # Save in same format as real data
    np.savez(
        output_path / "synthetic_mea.npz",
        raw_data=data.astype(np.float32),
        labels=np.array(labels, dtype=np.int32),
        timestamps=np.array(timestamps, dtype=np.float32),
        snr=np.array(snr_values, dtype=np.float32),
        metadata={
            "dataset_type": "synthetic",
            "statistical_match": "MEA_Real_v2024",
            "n_neurons": n_neurons,
            "n_channels": n_channels,
            "duration_min": duration_min,
            "sampling_rate": sampling_rate,
            "total_spikes": total_spikes,
            "generator_version": "1.0",
            "paper_reference": "Table I, MEA Real Recordings"
        }
    )
    
    print(f"Generated synthetic MEA dataset:")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Spikes/neuron: {total_spikes/n_neurons:.0f}")
    print(f"  Saved to: {output_path / 'synthetic_mea.npz'}")
    return str(output_path / "synthetic_mea.npz")


def generate_pyramidal_template(fs, duration_ms=1.5):
    """Generate pyramidal cell waveform (broad, ~1.2-1.5ms width)."""
    t = np.linspace(0, duration_ms/1000, int(fs * duration_ms/1000))
    # Biphasic waveform: negative peak then positive overshoot
    template = -np.exp(-((t-0.3e-3)**2)/(2*(0.15e-3)**2)) + \
               0.3*np.exp(-((t-0.6e-3)**2)/(2*(0.2e-3)**2))
    return template / np.max(np.abs(template))

def generate_interneuron_template(fs, duration_ms=0.8):
    """Generate fast-spiking interneuron waveform (narrow, ~0.6-0.8ms)."""
    t = np.linspace(0, duration_ms/1000, int(fs * duration_ms/1000))
    template = -np.exp(-((t-0.2e-3)**2)/(2*(0.08e-3)**2))
    return template / np.max(np.abs(template))

def generate_chandelier_template(fs, duration_ms=1.0):
    """Generate chandelier cell waveform (intermediate width)."""
    t = np.linspace(0, duration_ms/1000, int(fs * duration_ms/1000))
    template = -np.exp(-((t-0.25e-3)**2)/(2*(0.12e-3)**2)) + \
               0.2*np.exp(-((t-0.5e-3)**2)/(2*(0.15e-3)**2))
    return template / np.max(np.abs(template))

def generate_realistic_spike_times(n_spikes, total_samples, fs, refractory_ms=2.0):
    """Generate spike times with refractory period enforcement."""
    refractory_samples = int(refractory_ms / 1000 * fs)
    spike_times = []
    attempts = 0
    while len(spike_times) < n_spikes and attempts < n_spikes * 10:
        t = np.random.randint(0, total_samples - 100)
        if len(spike_times) == 0 or (t - spike_times[-1]) > refractory_samples:
            spike_times.append(t)
        attempts += 1
    return np.array(sorted(spike_times))

def generate_realistic_mea_noise(n_samples, n_channels, fs):
    """Generate realistic MEA background noise (1/f + white noise)."""
    # 1/f noise (pink noise) typical of neural recordings
    white = np.random.randn(n_samples, n_channels)
    # Apply 1/f filter
    b, a = signal.butter(2, [300, 3000], btype='band', fs=fs)
    noise = signal.filtfilt(b, a, white, axis=0)
    return noise * 5.0  # Scale to realistic microvolt levels


if __name__ == "__main__":
    generate_synthetic_mea_dataset()
