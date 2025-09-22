import pickle
import numpy as np
from pathlib import Path

def generate_cube_mic_positions(center_pos):
    """
    Generates 6 microphone positions centered around the given center position,
    corresponding to the middle of each face of a cube with a side length of 1 cm.
    """
    offsets = np.array([
        [0.005, 0, 0], [-0.005, 0, 0],  # Left/Right
        [0, 0.005, 0], [0, -0.005, 0],  # Front/Back
        [0, 0, 0.005], [0, 0, -0.005]   # Top/Bottom
    ])
    
    return [center_pos + offset for offset in offsets]

def update_mic_positions(input_pkl, output_pkl):
    """
    Updates the microphone positions in the given pkl file and saves the new file.
    """
    with open(input_pkl, "rb") as f:
        configs = pickle.load(f)
    
    for config in configs:
        center_pos = np.array(config["mic_pos"][0])  # First mic becomes cube center
        config["mic_pos"] = generate_cube_mic_positions(center_pos)
    
    with open(output_pkl, "wb") as f:
        pickle.dump(configs, f)

data_types = ["train", "validation", "test"]
config_dir = Path("configs")
for data_type in data_types:
    input_pkl = config_dir / f"MC_Libri_fixed_{data_type}.pkl"
    output_pkl = config_dir / f"MC_Libri_fixed_cube_{data_type}.pkl"
    update_mic_positions(input_pkl, output_pkl)