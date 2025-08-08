import pickle
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

def compute_angle_between_speakers_relative_to_mics(spk_pos, mic_pos):
    mic_center = np.mean(np.array(mic_pos), axis=0)
    vec1 = np.array(spk_pos[0]) - mic_center
    vec2 = np.array(spk_pos[1]) - mic_center
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def stratify_dataset_by_angle_overlap(input_pkl, output_pkl, angle_bins=9, overlap_bins=9):
    with open(input_pkl, "rb") as f:
        configs = pickle.load(f)

    total_samples = len(configs)
    print(f"Loaded {total_samples} samples from {input_pkl.name}")

    bin_dict = defaultdict(list)
    for cfg in configs:
        angle = compute_angle_between_speakers_relative_to_mics(cfg['spk_pos'], cfg['mic_pos'])
        overlap = cfg['overlap_ratio']
        angle_bin = int(np.clip(angle / (180 / angle_bins), 0, angle_bins - 1))
        overlap_bin = int(np.clip(overlap / (1.0 / overlap_bins), 0, overlap_bins - 1))
        bin_dict[(angle_bin, overlap_bin)].append(cfg)

    total_bins = angle_bins * overlap_bins
    all_bins = [(i, j) for i in range(angle_bins) for j in range(overlap_bins)]

    # Only keep non-empty bins
    non_empty_bins = [b for b in all_bins if len(bin_dict[b]) > 0]
    num_bins = len(non_empty_bins)

    samples_per_bin = total_samples // num_bins
    leftover = total_samples % num_bins

    print(f"Using {num_bins} non-empty bins")
    print(f"Target: {samples_per_bin} samples per bin x {num_bins} bins + {leftover} extra")

    selected_configs = []

    for idx, bin_key in enumerate(non_empty_bins):
        bin_samples = bin_dict[bin_key]
        target_samples = samples_per_bin + (1 if idx < leftover else 0)
        if len(bin_samples) >= target_samples:
            selected = random.sample(bin_samples, target_samples)
        else:
            selected = random.choices(bin_samples, k=target_samples)
        selected_configs.extend(selected)

    print(f"Generated stratified dataset with {len(selected_configs)} samples.")
    with open(output_pkl, "wb") as f:
        pickle.dump(selected_configs, f)


data_types = ["train", "validation", "test"]
config_dir = Path("configs")
for data_type in data_types:
    input_pkl = config_dir / f"MC_Libri_fixed_cube_{data_type}.pkl"
    output_pkl = config_dir / f"MC_Libri_fixed_cube_stratified_{data_type}.pkl"
    stratify_dataset_by_angle_overlap(input_pkl, output_pkl, angle_bins=9, overlap_bins=9)
