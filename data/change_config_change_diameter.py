import pickle
import numpy as np
from pathlib import Path

def scale_circle_mic_positions(original_mics, new_diameter=0.20):
    """
    Scales a circular 2D microphone array to a new diameter.
    Assumes all microphones lie in the same plane (same z), evenly distributed around a center.
    """
    original_mics = np.array(original_mics)
    mic_center = np.mean(original_mics, axis=0)
    radius = new_diameter / 2.0

    scaled_mics = []
    for mic in original_mics:
        vec = mic - mic_center
        vec[2] = 0  # Keep original height
        unit_vec = vec / np.linalg.norm(vec)
        new_mic = mic_center + radius * unit_vec
        new_mic[2] = mic[2]  # Restore original height
        scaled_mics.append(new_mic)
    return scaled_mics

def update_mic_positions(input_pkl, output_pkl, new_diameter=0.20):
    with open(input_pkl, "rb") as f:
        configs = pickle.load(f)

    for config in configs:
        config["mic_pos"] = scale_circle_mic_positions(config["mic_pos"], new_diameter)

    with open(output_pkl, "wb") as f:
        pickle.dump(configs, f)

# Process all data splits
data_types = ["train", "validation", "test"]
config_dir = Path("configs")
for data_type in data_types:
    input_pkl = config_dir / f"MC_Libri_fixed_{data_type}.pkl"
    output_pkl = config_dir / f"MC_Libri_fixed_circle20cm_{data_type}.pkl"
    update_mic_positions(input_pkl, output_pkl, new_diameter=0.20)
