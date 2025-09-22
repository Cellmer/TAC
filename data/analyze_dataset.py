import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def compute_angle_between_speakers_relative_to_mics(spk_pos, mic_pos):
    mic_center = np.mean(np.array(mic_pos), axis=0)
    vec1 = np.array(spk_pos[0]) - mic_center
    vec2 = np.array(spk_pos[1]) - mic_center
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def analyze_dataset(config_path, vmin=None, vmax=None):
    with open(config_path, 'rb') as f:
        configs = pickle.load(f)

    overlap_ratios, angles = [], []
    room_sizes, rt60s, spk_snrs, noise_snrs = [], [], [], []
    spk_heights_all = []
    mic_heights_all = []

    for cfg in configs:
        overlap_ratios.append(cfg['overlap_ratio'])
        angles.append(compute_angle_between_speakers_relative_to_mics(cfg['spk_pos'], cfg['mic_pos']))
        room_sizes.append(cfg['room_size'])
        rt60s.append(cfg['RT60'])
        spk_snrs.append(cfg['spk_snr'])
        noise_snrs.append(cfg['noise_snr'])
        spk_heights_all.extend([pos[2] for pos in cfg['spk_pos']])
        mic_heights_all.extend([pos[2] for pos in cfg['mic_pos']])

    room_sizes = np.array(room_sizes)
    room_widths, room_lengths, room_heights = room_sizes[:, 0], room_sizes[:, 1], room_sizes[:, 2]

    print(f"Liczba próbek: {len(configs)}")
    print(f"Współczynnik nakładania: średnia={np.mean(overlap_ratios):.3f}, min={np.min(overlap_ratios):.3f}, max={np.max(overlap_ratios):.3f}")
    print(f"Kąt między mówcami: średnia={np.mean(angles):.2f}°, min={np.min(angles):.2f}°, max={np.max(angles):.2f}°")
    print(f"Wymiary pomieszczenia (SxDxW):")
    print(f"  Szerokość:  średnia={np.mean(room_widths):.2f}, min={np.min(room_widths):.2f}, max={np.max(room_widths):.2f}")
    print(f"  Długość:    średnia={np.mean(room_lengths):.2f}, min={np.min(room_lengths):.2f}, max={np.max(room_lengths):.2f}")
    print(f"  Wysokość:   średnia={np.mean(room_heights):.2f}, min={np.min(room_heights):.2f}, max={np.max(room_heights):.2f}")
    print(f"RT60: średnia={np.mean(rt60s):.3f}, min={np.min(rt60s):.3f}, max={np.max(rt60s):.3f}")
    print(f"SNR mówcy: średnia={np.mean(spk_snrs):.2f}, min={np.min(spk_snrs):.2f}, max={np.max(spk_snrs):.2f}")
    print(f"SNR szumu: średnia={np.mean(noise_snrs):.2f}, min={np.min(noise_snrs):.2f}, max={np.max(noise_snrs):.2f}")
    print(f"Wysokości mówców (z): średnia={np.mean(spk_heights_all):.3f} m, min={np.min(spk_heights_all):.3f} m, max={np.max(spk_heights_all):.3f} m")
    print(f"Wysokości mikrofonów (z): średnia={np.mean(mic_heights_all):.3f} m, min={np.min(mic_heights_all):.3f} m, max={np.max(mic_heights_all):.3f} m")

    base_name = os.path.splitext(os.path.basename(config_path))[0]

    def save_hist(data, title, xlabel, filename, bins=30):
        plt.figure()
        plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Liczba próbek')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{base_name}_{filename}.png")
        plt.close()
        print(f"Zapisano: {base_name}_{filename}.png")

    # Histogramy
    save_hist(overlap_ratios, 'Rozkład współczynnika nakładania', 'Współczynnik nakładania', 'overlap_ratio')
    save_hist(angles, 'Rozkład kąta między mówcami', 'Kąt (stopnie)', 'speaker_angle')
    save_hist(room_widths, 'Rozkład szerokości pomieszczeń', 'Szerokość (m)', 'room_width')
    save_hist(room_lengths, 'Rozkład długości pomieszczeń', 'Długość (m)', 'room_length')
    save_hist(room_heights, 'Rozkład wysokości pomieszczeń', 'Wysokość (m)', 'room_height')
    save_hist(rt60s, 'Rozkład czasu pogłosu (RT60)', 'RT60 (s)', 'rt60')
    save_hist(spk_snrs, 'Rozkład SNR mówcy', 'SNR (dB)', 'spk_snr')
    save_hist(noise_snrs, 'Rozkład SNR szumu', 'SNR (dB)', 'noise_snr')
    save_hist(spk_heights_all, 'Rozkład wysokości mówców (z)', 'Wysokość mówcy (m)', 'speaker_height')
    save_hist(mic_heights_all, 'Rozkład wysokości mikrofonów (z)', 'Wysokość mikrofonu (m)', 'mic_height')

    # Heatmap: kąt vs współczynnik nakładania
    df = pd.DataFrame({
        'angle': angles,
        'overlap': overlap_ratios
    })

    angle_edges = np.linspace(0, 180, 10)
    overlap_edges = np.linspace(0, 1, 10)
    angle_labels = [f"({int(angle_edges[i])}°, {int(angle_edges[i+1])}°]" for i in range(len(angle_edges)-1)]
    overlap_labels = [f"[{overlap_edges[i]:.1f}, {overlap_edges[i+1]:.1f})" for i in range(len(overlap_edges)-2)] \
                     + [f"[{overlap_edges[-2]:.1f}, {overlap_edges[-1]:.1f}]"]

    df['angle_bin'] = pd.cut(df['angle'], bins=angle_edges, labels=angle_labels)
    df['overlap_bin'] = pd.cut(df['overlap'], bins=overlap_edges, labels=overlap_labels, include_lowest=True)

    heatmap_data = df.pivot_table(index='angle_bin', columns='overlap_bin', aggfunc='size', fill_value=0)
    heatmap_data = heatmap_data.reindex(index=angle_labels[::-1], columns=overlap_labels)

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='d',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Liczba próbek'}
    )
    ax = plt.gca()
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label('Liczba próbek', size=14)
            cbar.ax.tick_params(labelsize=12)

    plt.title('Liczba próbek w przedziałach (kąt, współczynnik nakładania)')
    plt.xlabel("Przedział współczynnika nakładania")
    plt.ylabel("Przedział kąta (stopnie)")
    plt.tight_layout()
    heatmap_path = f"{base_name}_angle_vs_overlap_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"Zapisano: {heatmap_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analiza statystyk zbioru danych z pliku konfiguracyjnego.')
    parser.add_argument('--config', type=str, required=True,
                        help='Ścieżka do pliku konfiguracyjnego zbioru danych (.pkl)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Dolna granica skali kolorów dla heatmapy')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Górna granica skali kolorów dla heatmapy')
    args = parser.parse_args()
    analyze_dataset(args.config, vmin=args.vmin, vmax=args.vmax)
