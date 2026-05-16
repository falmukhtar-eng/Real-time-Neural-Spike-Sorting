"""
SpikeForest-compatible wrapper for CNN-BiLSTM.
This allows direct comparison with Kilosort3, YASS, IronClust on standard benchmarks.

Usage:
    python benchmarks/spikeforest_wrapper.py --recording data/test_recording --output sorting_output
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import your model
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from model import build_model
from preprocess import preprocess_recording

class CNNBiLSTMSorter:
    """
    Standard interface for spike sorting benchmarks.
    Input: Raw recording (numpy array or SpikeInterface RecordingExtractor)
    Output: Spike times and unit IDs
    """
    
    def __init__(self, model_weights, num_classes=6, num_pca=47):
        self.model = build_model(num_classes=num_classes, num_pca_components=num_pca)
        self.model.load_weights(model_weights)
        self.num_classes = num_classes
        
    def run(self, recording, sampling_rate=24000):
        """
        Run sorting on a recording.
        
        Parameters:
        -----------
        recording : np.ndarray
            Raw neural recording, shape (n_samples, n_channels) or (n_samples,)
        sampling_rate : int
            Sampling frequency in Hz
            
        Returns:
        --------
        sorting : dict
            {unit_id: spike_times_array}
        """
        # Step 1: Preprocess (bandpass, detect, extract waveforms, PCA)
        preprocessed = preprocess_recording(
            recording, 
            sampling_rate=sampling_rate,
            pca_variance=0.952
        )
        
        # Step 2: Run CNN-BiLSTM inference
        predictions = self.model.predict(preprocessed["waveforms"])
        
        # Step 3: Convert predictions to spike times
        sorting = {}
        for unit_id in range(self.num_classes):
            mask = np.argmax(predictions["classification"], axis=1) == unit_id
            spike_times = preprocessed["timestamps"][mask]
            sorting[f"unit_{unit_id}"] = spike_times
            
        return sorting


def main():
    parser = argparse.ArgumentParser(description="Run CNN-BiLSTM on a recording")
    parser.add_argument("--recording", required=True, help="Path to recording file (.npy or .bin)")
    parser.add_argument("--model-weights", default="models/pretrained/quiroga_full_supervision.h5")
    parser.add_argument("--output", required=True, help="Output directory for sorting results")
    parser.add_argument("--sampling-rate", type=int, default=24000)
    args = parser.parse_args()
    
    # Load recording
    recording = np.load(args.recording)
    
    # Run sorter
    sorter = CNNBiLSTMSorter(args.model_weights)
    sorting = sorter.run(recording, sampling_rate=args.sampling_rate)
    
    # Save results in standard format
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for unit_id, spike_times in sorting.items():
        np.save(output_dir / f"{unit_id}_spike_times.npy", spike_times)
    
    print(f"Sorting complete. Results saved to {output_dir}")
    print(f"Units found: {len(sorting)}")
    for unit_id, spike_times in sorting.items():
        print(f"  {unit_id}: {len(spike_times)} spikes")

if __name__ == "__main__":
    main()
