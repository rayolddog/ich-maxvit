"""
Experiment: Fixed HU Window vs. Per-Image Normalization

For 1000 ICH-positive CT slices, compare how three normalization schemes
encode gray matter and acute blood pixels into tensor space.

Source: /home/johnb/cache_1ch  (pre-built wide-window cache)
  Stored encoding: image_norm = (HU + 300) / 480, clipped to [-300, 180] HU
  HU recovery:     HU = image_norm * 480 - 300

Ground truth tissue labels are derived purely from HU values — no manual
segmentation needed.  The same physical pixels are evaluated under all schemes.

  Gray matter  :  HU ∈ [28, 42]   (cortical gray matter)
  Acute blood  :  HU ∈ [50, 80]   (acute ICH; hyperacute excluded)

Normalization schemes compared:
  1. Per-image  : tensor = (HU − slice_min) / (slice_max − slice_min)
                  Standard image preprocessing — destroys HU calibration.
  2. Fixed medium: tensor = clip((HU + 200) / 400, 0, 1)   window −200 to 200
  3. Fixed narrow: tensor = clip((HU −  48) /  42, 0, 1)   window   48 to  90

Output:
  normalization_comparison.png  — violin plots with Δmean and overlap index
  normalization_comparison.json — summary statistics

Usage:
  python compare_normalization.py \
      --cache-dir /home/johnb/cache_1ch \
      --labels    /home/johnb/Documents/stage_2_train.csv \
      --n 1000 --workers 8
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Tissue HU ranges (ground truth) ──────────────────────────────────────────
GM_HU_LOW,  GM_HU_HIGH  =  28.0,  42.0   # gray matter
BLD_HU_LOW, BLD_HU_HIGH =  50.0,  80.0   # acute ICH blood

# ── cache_1ch encoding: image_norm = (HU + 300) / 480  clipped [-300, 180] HU
CACHE_OFFSET = 300.0
CACHE_RANGE  = 480.0

# ── Normalization schemes ─────────────────────────────────────────────────────
SCHEMES = {
    'Per-image\n(min/max)':          None,           # computed per slice
    'Fixed medium\n(−200 to 200 HU)': (-200.0, 200.0),
    'Fixed narrow\n(48 to 90 HU)':    (  48.0,  90.0),
}

MIN_PIXELS = 20   # minimum tissue pixels per class to include a slice

LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular',
              'subarachnoid', 'subdural', 'any']


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_one(args_tuple):
    """
    Worker: load one cache_1ch .npz, recover HU, return tissue pixel values
    under each normalization scheme.

    Returns dict {scheme_name: {'gm': array, 'blood': array}} or None.
    """
    npz_path, = args_tuple

    try:
        data = np.load(npz_path)
        image_norm = data['image_norm'].astype(np.float32)
    except Exception:
        return None

    # Recover HU from cache_1ch encoding
    hu = image_norm * CACHE_RANGE - CACHE_OFFSET

    # Tissue masks from HU (ground truth — same pixels evaluated in all schemes)
    gm_mask  = (hu >= GM_HU_LOW)  & (hu <= GM_HU_HIGH)
    bld_mask = (hu >= BLD_HU_LOW) & (hu <= BLD_HU_HIGH)

    if gm_mask.sum() < MIN_PIXELS or bld_mask.sum() < MIN_PIXELS:
        return None

    results = {}
    for scheme_name, params in SCHEMES.items():
        if params is None:
            # Per-image: normalise from this slice's own HU min/max
            hu_min   = hu.min()
            hu_max   = hu.max()
            hu_range = hu_max - hu_min
            if hu_range < 1.0:
                continue
            tensor = (hu - hu_min) / hu_range
        else:
            lo, hi = params
            tensor = np.clip((hu - lo) / (hi - lo), 0.0, 1.0)

        results[scheme_name] = {
            'gm':    tensor[gm_mask].astype(np.float16),
            'blood': tensor[bld_mask].astype(np.float16),
        }

    return results if results else None


# ── Statistics ────────────────────────────────────────────────────────────────

def overlap_index(a: np.ndarray, b: np.ndarray, bins: int = 300) -> float:
    """
    Histogram overlap index (Bhattacharyya-like).
    0.0 = no overlap (perfect separation), 1.0 = identical distributions.
    Lower is better for tissue discrimination.
    """
    lo  = min(float(a.min()), float(b.min()))
    hi  = max(float(a.max()), float(b.max()))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb)) * width)


def summarize(gm: np.ndarray, bld: np.ndarray) -> dict:
    return {
        'gm_mean':        float(gm.mean()),
        'gm_std':         float(gm.std()),
        'gm_median':      float(np.median(gm)),
        'blood_mean':     float(bld.mean()),
        'blood_std':      float(bld.std()),
        'blood_median':   float(np.median(bld)),
        'delta_mean':     float(abs(bld.mean() - gm.mean())),
        'overlap_index':  overlap_index(gm, bld),
        'n_gm_pixels':    int(len(gm)),
        'n_blood_pixels': int(len(bld)),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(collected: dict, stats: dict, n_slices: int, out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scheme_names = list(SCHEMES.keys())
    n_schemes    = len(scheme_names)

    fig, axes = plt.subplots(1, n_schemes, figsize=(5 * n_schemes, 7), sharey=True)
    fig.suptitle(
        'Tensor value distributions: Gray Matter vs. Acute Blood\n'
        f'Fixed HU window vs. per-image normalization  '
        f'(n={n_slices:,} ICH-positive slices)',
        fontsize=13, y=1.01)

    gm_color  = '#4878CF'
    bld_color = '#D65F5F'

    for ax, scheme in zip(axes, scheme_names):
        data    = collected.get(scheme, {})
        gm_all  = np.concatenate(data.get('gm',    [np.zeros(1)])).astype(float)
        bld_all = np.concatenate(data.get('blood',  [np.zeros(1)])).astype(float)

        vp = ax.violinplot(
            [gm_all, bld_all],
            positions=[1, 2],
            showmedians=True,
            showextrema=False,
            widths=0.7,
        )
        for body, color in zip(vp['bodies'], [gm_color, bld_color]):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(2.5)

        s = stats.get(scheme.replace('\n', ' '), {})
        delta   = s.get('delta_mean', 0.0)
        overlap = s.get('overlap_index', 1.0)

        ax.set_title(scheme, fontsize=11, pad=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Gray\nMatter\n(HU 28–42)',
                            'Acute\nBlood\n(HU 50–80)'], fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Tensor value [0, 1]', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.3, 2.7)
        ax.grid(axis='y', alpha=0.3)

        # Annotate delta mean and overlap index
        color = '#2d6a2d' if delta > 0.1 else ('#b8860b' if delta > 0.03 else '#8b0000')
        ax.text(0.5, 0.03,
                f"Δ mean = {delta:.3f}\nOverlap = {overlap:.3f}",
                transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color=color,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85,
                          edgecolor=color, linewidth=1.2))

        # Mean markers
        ax.scatter([1, 2], [gm_all.mean(), bld_all.mean()],
                   marker='D', s=40, color='white', zorder=5)

    legend = [Patch(fc=gm_color,  label='Gray matter  (HU 28–42)'),
              Patch(fc=bld_color, label='Acute blood  (HU 50–80)')]
    fig.legend(handles=legend, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=11,
               framealpha=0.9)

    # Footnote
    fig.text(0.5, -0.08,
             'Overlap index: 0.0 = perfect separation, 1.0 = identical distributions. '
             'Lower overlap and higher Δ mean indicate better tissue discrimination.',
             ha='center', fontsize=8, style='italic', color='#444444')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare HU normalization schemes on ICH-positive CT slices')
    parser.add_argument('--cache-dir', default='/home/johnb/cache_1ch')
    parser.add_argument('--labels',    default='/home/johnb/Documents/stage_2_train.csv')
    parser.add_argument('--n',         type=int, default=1000,
                        help='ICH-positive slices to sample (default: 1000)')
    parser.add_argument('--workers',   type=int, default=8)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--out-plot',  default='normalization_comparison.png')
    parser.add_argument('--out-json',  default='normalization_comparison.json')
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load labels, filter to ICH-positive ──────────────────────────────────
    print(f"Loading labels: {args.labels}")
    df = pd.read_csv(args.labels)

    # Format: ID_<image_id>_<hemorrhage_type>, Label
    df['image_id']    = df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['label_type']  = df['ID'].apply(lambda x: x.split('_')[2] if x.count('_') >= 2 else x)

    pivot = df.pivot_table(index='image_id', columns='label_type',
                           values='Label', aggfunc='first')
    any_col = 'any' if 'any' in pivot.columns else pivot.columns[-1]
    ich_pos_ids = set(pivot[pivot[any_col] == 1].index.tolist())
    print(f"  ICH-positive IDs in labels: {len(ich_pos_ids):,}")

    # ── Find matching cache files ─────────────────────────────────────────────
    print(f"Scanning cache: {args.cache_dir}")
    available = set()
    with os.scandir(args.cache_dir) as it:
        for entry in it:
            if entry.name.endswith('.npz') and entry.is_file():
                available.add(entry.name[:-4])

    candidates = list(ich_pos_ids & available)
    print(f"  Cached ICH-positive slices: {len(candidates):,}")

    if len(candidates) == 0:
        print("ERROR: no matching cached files found.")
        return

    sample_n = min(args.n, len(candidates))
    sampled  = random.sample(candidates, sample_n)
    print(f"  Sampling {sample_n:,} slices (seed={args.seed})")

    # ── Process ───────────────────────────────────────────────────────────────
    scheme_names = list(SCHEMES.keys())
    collected    = {s: {'gm': [], 'blood': []} for s in scheme_names}

    n_ok     = 0
    n_skip   = 0

    npz_paths = [os.path.join(args.cache_dir, f"{sid}.npz") for sid in sampled]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, (p,)): p for p in npz_paths}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                n_skip += 1
                continue
            n_ok += 1
            for scheme, tissues in result.items():
                collected[scheme]['gm'].append(tissues['gm'].astype(np.float32))
                collected[scheme]['blood'].append(tissues['blood'].astype(np.float32))
            if n_ok % 100 == 0:
                print(f"  {n_ok} slices processed  ({n_skip} skipped — "
                      f"insufficient tissue pixels)", flush=True)

    print(f"\nDone: {n_ok} usable slices, "
          f"{n_skip} skipped (< {MIN_PIXELS} tissue pixels)")

    # ── Summarize ─────────────────────────────────────────────────────────────
    print(f"\n{'Scheme':<34}  {'GM mean±std':>15}  {'Blood mean±std':>15}  "
          f"{'Δmean':>7}  {'Overlap':>8}")
    print("─" * 86)

    all_stats = {}
    for scheme in scheme_names:
        gm_all  = np.concatenate(collected[scheme]['gm'])   if collected[scheme]['gm']   else np.zeros(1)
        bld_all = np.concatenate(collected[scheme]['blood']) if collected[scheme]['blood'] else np.zeros(1)
        s = summarize(gm_all, bld_all)
        key = scheme.replace('\n', ' ')
        all_stats[key] = s

        print(f"{key:<34}  "
              f"{s['gm_mean']:.3f} ± {s['gm_std']:.3f}       "
              f"{s['blood_mean']:.3f} ± {s['blood_std']:.3f}  "
              f"{s['delta_mean']:>7.3f}  {s['overlap_index']:>8.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'n_slices_used': n_ok,
        'n_slices_skipped': n_skip,
        'tissue_ranges': {
            'gray_matter_hu':  [GM_HU_LOW,  GM_HU_HIGH],
            'acute_blood_hu':  [BLD_HU_LOW, BLD_HU_HIGH],
        },
        'schemes': all_stats,
    }
    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nStatistics → {args.out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_plot(collected, all_stats, n_ok, args.out_plot)


if __name__ == '__main__':
    main()
