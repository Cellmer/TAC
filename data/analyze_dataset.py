import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_angle_between_speakers_relative_to_mics(spk_pos, mic_pos):
    mic_center = np.mean(np.array(mic_pos), axis=0)
    vec1 = np.array(spk_pos[0]) - mic_center
    vec2 = np.array(spk_pos[1]) - mic_center
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def analyze_dataset(config_path):
    with open(config_path, 'rb') as f:
        configs = pickle.load(f)

    overlap_ratios, angles = [], []
    room_sizes, rt60s, spk_snrs, noise_snrs = [], [], [], []

    for cfg in configs:
        overlap_ratios.append(cfg['overlap_ratio'])
        angles.append(compute_angle_between_speakers_relative_to_mics(cfg['spk_pos'], cfg['mic_pos']))
        room_sizes.append(cfg['room_size'])
        rt60s.append(cfg['RT60'])
        spk_snrs.append(cfg['spk_snr'])
        noise_snrs.append(cfg['noise_snr'])

    room_sizes = np.array(room_sizes)
    room_widths, room_lengths, room_heights = room_sizes[:, 0], room_sizes[:, 1], room_sizes[:, 2]

    print(f"Total samples: {len(configs)}")
    print(f"Overlap Ratio: mean={np.mean(overlap_ratios):.3f}, min={np.min(overlap_ratios):.3f}, max={np.max(overlap_ratios):.3f}")
    print(f"Speaker Angle: mean={np.mean(angles):.2f}°, min={np.min(angles):.2f}°, max={np.max(angles):.2f}°")
    print(f"Room Size (WxLxH): Width={np.mean(room_widths):.2f}, Length={np.mean(room_lengths):.2f}, Height={np.mean(room_heights):.2f}")
    print(f"RT60: mean={np.mean(rt60s):.3f}, spk_snr: mean={np.mean(spk_snrs):.2f}, noise_snr: mean={np.mean(noise_snrs):.2f}")

    base_name = os.path.splitext(os.path.basename(config_path))[0]

    def save_hist(data, title, xlabel, filename, bins=30):
        plt.figure()
        plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{base_name}_{filename}.png")
        plt.close()
        print(f"Saved: {base_name}_{filename}.png")

    # Histograms
    save_hist(overlap_ratios, 'Overlap Ratio Distribution', 'Overlap Ratio', 'overlap_ratio')
    save_hist(angles, 'Angle Between Speakers Distribution', 'Angle (degrees)', 'speaker_angle')
    save_hist(room_widths, 'Room Width Distribution', 'Width (m)', 'room_width')
    save_hist(room_lengths, 'Room Length Distribution', 'Length (m)', 'room_length')
    save_hist(room_heights, 'Room Height Distribution', 'Height (m)', 'room_height')
    save_hist(rt60s, 'RT60 Distribution', 'RT60 (s)', 'rt60')
    save_hist(spk_snrs, 'Speaker SNR Distribution', 'SNR (dB)', 'spk_snr')
    save_hist(noise_snrs, 'Noise SNR Distribution', 'SNR (dB)', 'noise_snr')

    # Heatmap of Angle vs Overlap Ratio
    import pandas as pd

    df = pd.DataFrame({
        'angle': angles,
        'overlap': overlap_ratios
    })

    angle_edges = np.linspace(0, 180, 10)
    overlap_edges = np.linspace(0, 1, 10)
    angle_labels = [f"({int(angle_edges[i])}°, {int(angle_edges[i+1])}°]" for i in range(len(angle_edges)-1)]
    overlap_labels = [f"[{overlap_edges[i]:.1f}, {overlap_edges[i+1]:.1f})" for i in range(len(overlap_edges)-2)] + [f"[{overlap_edges[-2]:.1f}, {overlap_edges[-1]:.1f}]"]

    df['angle_bin'] = pd.cut(df['angle'], bins=angle_edges, labels=angle_labels)
    df['overlap_bin'] = pd.cut(df['overlap'], bins=overlap_edges, labels=overlap_labels, include_lowest=True)

    heatmap_data = df.pivot_table(index='angle_bin', columns='overlap_bin', aggfunc='size', fill_value=0)
    heatmap_data = heatmap_data.sort_index(ascending=False)  # low angles at bottom

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='viridis')
    plt.title('Sample Count per (Angle, Overlap Ratio) Bin')
    plt.xlabel("Overlap Ratio Bin")
    plt.ylabel("Angle Bin (degrees)")
    plt.tight_layout()
    heatmap_path = f"{base_name}_angle_vs_overlap_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved: {heatmap_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze dataset config statistics.')
    parser.add_argument('--config', type=str, required=True, help='Path to the dataset config .pkl file')
    args = parser.parse_args()
    analyze_dataset(args.config)
