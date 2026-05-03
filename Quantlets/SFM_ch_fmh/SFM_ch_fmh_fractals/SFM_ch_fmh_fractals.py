"""
SFM_ch_fmh_fractals
====================
Fractal Market Hypothesis: visual gallery of fractals.

Generates publication-quality figures for the FMH chapter:
  - Mandelbrot set (full view + 3 zoom levels)
  - Julia sets (4 different parameters c)
  - Barnsley fern (IFS)
  - Sierpinski triangle (IFS / chaos game)
  - Koch snowflake (L-system)
  - Dragon curve (L-system)
  - Cantor set
  - Fractal coastline (random midpoint displacement)
  - Fractal tree (recursive)
  - Fractional Brownian motion (4 H values)
  - Box-counting dimension illustration

All output goes to ../../charts/ as PDF + PNG (transparent background).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection

# Standard chart style (presentation-quality, large + transparent)
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 120

CHART_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'charts'))
os.makedirs(CHART_DIR, exist_ok=True)

MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Orange   = '#E67E22'
Purple   = '#8E44AD'
Amber    = '#B5853F'

def save_fig(name):
    pdf_path = os.path.join(CHART_DIR, f'{name}.pdf')
    png_path = os.path.join(CHART_DIR, f'{name}.png')
    plt.savefig(pdf_path, bbox_inches='tight', transparent=True)
    plt.savefig(png_path, bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf / .png")


# =============================================================================
# 1. MANDELBROT SET
# =============================================================================
def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=256):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.full(C.shape, max_iter, dtype=float)
    mask = np.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        diverged = np.abs(Z) > 2
        new_div = diverged & mask
        # smooth coloring
        if new_div.any():
            zn = np.abs(Z[new_div])
            M[new_div] = i + 1 - np.log(np.log(zn)) / np.log(2)
        mask &= ~diverged
    M[mask] = np.nan  # interior
    return M


def plot_mandelbrot_gallery():
    print("1. MANDELBROT SET (gallery)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    views = [
        ('A: Global view', -2.2, 1.0, -1.3, 1.3, 800, 700, 200),
        ('B: Seahorse Valley', -0.78, -0.72, 0.07, 0.13, 700, 700, 400),
        ('C: Elephant Valley', 0.245, 0.275, -0.015, 0.015, 700, 700, 500),
        ('D: Mini-Mandelbrot', -1.78, -1.72, -0.03, 0.03, 700, 700, 500),
    ]
    cmap = colors.LinearSegmentedColormap.from_list(
        'mandel', ['#0a0a2a', MainBlue, '#3a7ec0', '#f0d080',
                   Crimson, '#5a0010', 'black'])

    for ax, (title, xmin, xmax, ymin, ymax, w, h, it) in zip(axes.ravel(), views):
        M = mandelbrot(xmin, xmax, ymin, ymax, w, h, it)
        ax.imshow(M, cmap=cmap, extent=[xmin, xmax, ymin, ymax],
                  origin='lower', interpolation='bilinear')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    plt.tight_layout()
    save_fig('ch_fmh_mandelbrot_gallery')


def plot_mandelbrot_single():
    print("2. MANDELBROT SET (large single)")
    fig, ax = plt.subplots(figsize=(10, 8))
    M = mandelbrot(-2.2, 1.0, -1.25, 1.25, 1200, 900, 300)
    cmap = colors.LinearSegmentedColormap.from_list(
        'mandel', ['#0a0a2a', MainBlue, '#3a7ec0', '#f0d080',
                   Crimson, '#5a0010', 'black'])
    ax.imshow(M, cmap=cmap, extent=[-2.2, 1.0, -1.25, 1.25],
              origin='lower', interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    save_fig('ch_fmh_mandelbrot_full')


# =============================================================================
# 2. JULIA SETS
# =============================================================================
def julia(c, xmin, xmax, ymin, ymax, width, height, max_iter=256):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    M = np.full(Z.shape, max_iter, dtype=float)
    mask = np.ones(Z.shape, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        diverged = np.abs(Z) > 2
        new_div = diverged & mask
        if new_div.any():
            zn = np.abs(Z[new_div])
            M[new_div] = i + 1 - np.log(np.log(zn)) / np.log(2)
        mask &= ~diverged
    M[mask] = np.nan
    return M


def plot_julia_gallery():
    print("3. JULIA SETS (gallery)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    julia_params = [
        (complex(-0.7, 0.27015), 'A: $c=-0.7+0.27015i$ (Dendrite-like)'),
        (complex(-0.8,  0.156),  'B: $c=-0.8+0.156i$ (Douady\'s rabbit)'),
        (complex( 0.285, 0.01),  'C: $c=0.285+0.01i$ (Connected)'),
        (complex(-0.4,  0.6),    'D: $c=-0.4+0.6i$ (Spiral structure)'),
    ]
    cmap = colors.LinearSegmentedColormap.from_list(
        'julia', ['#0a1a2a', Purple, '#c060c0', '#fff0c0',
                  Orange, Crimson, 'black'])

    for ax, (c, title) in zip(axes.ravel(), julia_params):
        J = julia(c, -1.6, 1.6, -1.2, 1.2, 800, 700, 300)
        ax.imshow(J, cmap=cmap, extent=[-1.6, 1.6, -1.2, 1.2],
                  origin='lower', interpolation='bilinear')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    plt.tight_layout()
    save_fig('ch_fmh_julia_gallery')


# =============================================================================
# 3. BARNSLEY FERN  (IFS that mimics a real fern)
# =============================================================================
def plot_barnsley_fern(n_points=600000):
    print("4. BARNSLEY FERN (density-rendered)")
    # IFS transformations (a, b, c, d, e, f, p)
    transforms = np.array([
        [ 0.00,  0.00,  0.00,  0.16, 0.00, 0.00, 0.01],
        [ 0.85,  0.04, -0.04,  0.85, 0.00, 1.60, 0.85],
        [ 0.20, -0.26,  0.23,  0.22, 0.00, 1.60, 0.07],
        [-0.15,  0.28,  0.26,  0.24, 0.00, 0.44, 0.07],
    ])
    probs = transforms[:, 6]
    cum = np.cumsum(probs)

    x, y = 0.0, 0.0
    pts_x = np.empty(n_points); pts_y = np.empty(n_points)
    rng = np.random.default_rng(42)
    rs = rng.random(n_points)
    for i in range(n_points):
        r = rs[i]
        k = np.searchsorted(cum, r)
        a, b, c, d, e, f, _ = transforms[k]
        x, y = a * x + b * y + e, c * x + d * y + f
        pts_x[i] = x; pts_y[i] = y

    # Density-based rendering: 2D histogram with log intensity (research-grade)
    fig, ax = plt.subplots(figsize=(6, 9))
    nbins = 600
    H, xedges, yedges = np.histogram2d(pts_x, pts_y, bins=nbins,
                                       range=[[-3, 3], [0, 11]])
    H_log = np.log10(H + 1)
    ax.imshow(H_log.T, origin='lower', extent=[-3, 3, 0, 11],
              cmap='Greys', aspect='equal', interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.02, 0.98, r'$f_1: \text{Tulpina}\;(p{=}1\%)$' + '\n' +
            r'$f_2: \text{Frunza}\;(p{=}85\%)$' + '\n' +
            r'$f_3: \text{Dreapta}\;(p{=}7\%)$' + '\n' +
            r'$f_4: \text{Stanga}\;(p{=}7\%)$',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='none',
                      edgecolor='#888888', alpha=0.7, linewidth=0.4))
    plt.tight_layout()
    save_fig('ch_fmh_barnsley_fern')


# =============================================================================
# 4. SIERPINSKI TRIANGLE (chaos game)
# =============================================================================
def plot_sierpinski(n_points=80000):
    print("5. SIERPINSKI TRIANGLE")
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    rng = np.random.default_rng(7)
    pts = np.empty((n_points, 2))
    p = np.array([0.5, 0.3])
    idx = rng.integers(0, 3, n_points)
    for i in range(n_points):
        v = vertices[idx[i]]
        p = (p + v) / 2.0
        pts[i] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05, color=MainBlue, alpha=0.8, marker='.')
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    save_fig('ch_fmh_sierpinski')


# =============================================================================
# 5. KOCH SNOWFLAKE
# =============================================================================
def koch_curve(p1, p2, depth):
    if depth == 0:
        return [p1, p2]
    p1 = np.asarray(p1); p2 = np.asarray(p2)
    s = p1 + (p2 - p1) / 3.0
    e = p1 + 2 * (p2 - p1) / 3.0
    # rotate (e - s) by -60 degrees around s for the peak
    angle = -np.pi / 3
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    peak = s + R @ (e - s)
    return (koch_curve(p1, s, depth - 1)[:-1] +
            koch_curve(s, peak, depth - 1)[:-1] +
            koch_curve(peak, e, depth - 1)[:-1] +
            koch_curve(e, p2, depth - 1))


def plot_koch_snowflake():
    print("6. KOCH SNOWFLAKE")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    # Initial triangle
    h = np.sqrt(3) / 2
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, h])

    for ax, depth in zip(axes, [0, 1, 2, 4]):
        seg1 = koch_curve(A, B, depth)
        seg2 = koch_curve(B, C, depth)
        seg3 = koch_curve(C, A, depth)
        pts = seg1 + seg2[1:] + seg3[1:]
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=MainBlue, lw=0.8)
        ax.fill(pts[:, 0], pts[:, 1], color=MainBlue, alpha=0.10)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'Iteration {depth}', fontweight='bold', fontsize=10)

    plt.tight_layout()
    save_fig('ch_fmh_koch_snowflake')


# =============================================================================
# 6. DRAGON CURVE (L-system)
# =============================================================================
def dragon_points(iterations):
    seq = [1]
    for _ in range(iterations):
        new = list(seq)
        new.append(1)
        new.extend(-s for s in reversed(seq))
        seq = new
    angle = 0.0
    x, y = 0.0, 0.0
    pts = [(x, y)]
    step = 1.0
    for s in seq:
        x += step * np.cos(angle)
        y += step * np.sin(angle)
        pts.append((x, y))
        angle += s * np.pi / 2
    return np.array(pts)


def plot_dragon_curve():
    print("7. DRAGON CURVE")
    fig, ax = plt.subplots(figsize=(9, 7))
    pts = dragon_points(14)
    # color along path
    n = len(pts) - 1
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cmap = plt.cm.viridis
    cols = cmap(np.linspace(0, 1, n))
    lc = LineCollection(segs, colors=cols, linewidths=0.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    save_fig('ch_fmh_dragon_curve')


# =============================================================================
# 7. CANTOR SET
# =============================================================================
def plot_cantor():
    print("8. CANTOR SET")
    fig, ax = plt.subplots(figsize=(11, 5))

    def cantor_iter(intervals):
        new = []
        for a, b in intervals:
            third = (b - a) / 3
            new.append((a, a + third))
            new.append((b - third, b))
        return new

    intervals = [(0.0, 1.0)]
    levels = 7
    height = 0.6
    for k in range(levels):
        y = -k * height
        for a, b in intervals:
            ax.add_patch(plt.Rectangle((a, y), b - a, height * 0.85,
                                       facecolor=MainBlue, edgecolor='none'))
        ax.text(-0.04, y + height * 0.4, f'$C_{{{k}}}$', ha='right', va='center',
                fontsize=10, color=MainBlue)
        if k < levels - 1:
            intervals = cantor_iter(intervals)

    ax.set_xlim(-0.07, 1.02)
    ax.set_ylim(-(levels - 1) * height - 0.2, height + 0.2)
    ax.set_aspect('auto')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    save_fig('ch_fmh_cantor')


# =============================================================================
# 8. FRACTAL COASTLINE (random midpoint displacement)
# =============================================================================
def midpoint_displacement(n_iter, roughness, seed=1):
    rng = np.random.default_rng(seed)
    pts = np.array([0.0, 0.0])  # 1D heights
    pts = np.array([0.0, 1.0, 0.5, 1.5, 0.8, 0.0])
    # use generic 1D MD
    arr = np.array([0.0, 0.0])
    n = 1 + 2 ** n_iter
    arr = np.zeros(n)
    arr[0] = 0.0
    arr[-1] = 0.0
    step = (n - 1) // 2
    scale = 1.0
    while step >= 1:
        for i in range(step, n, 2 * step):
            mid = 0.5 * (arr[i - step] + arr[i + step])
            arr[i] = mid + rng.normal(0, scale)
        scale *= roughness
        step //= 2
    return arr


def plot_coastline():
    print("9. FRACTAL COASTLINE (3 H values)")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    Hs = [0.3, 0.5, 0.8]
    labels = ['H = 0.3 (anti-persistent, jagged coastline)',
              'H = 0.5 (random walk, neutral coastline)',
              'H = 0.8 (persistent, smooth coastline)']
    colors_h = [Crimson, MainBlue, Forest]

    for ax, H, lab, col in zip(axes, Hs, labels, colors_h):
        rough = 2 ** (-H)
        np.random.seed(int(H * 100))
        y = midpoint_displacement(10, rough, seed=int(H * 100))
        x = np.linspace(0, 1, len(y))
        baseline = y.min() - 0.3
        ax.fill_between(x, baseline, y, color=col, alpha=0.55)
        ax.plot(x, y, color=col, lw=0.6)
        ax.set_ylabel('Elevation')
        ax.set_title(lab, fontweight='bold', fontsize=9, loc='left',
                     color=col)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[-1].set_xticks([]); axes[-1].set_xlabel('Distance')
    plt.tight_layout()
    save_fig('ch_fmh_coastline')


# =============================================================================
# 9. FRACTAL TREE (recursive)
# =============================================================================
def plot_fractal_tree():
    print("10. FRACTAL TREE")
    fig, ax = plt.subplots(figsize=(8, 9))

    rng = np.random.default_rng(3)

    def branch(x, y, angle, length, depth, lw):
        if depth == 0 or length < 0.005:
            return
        x2 = x + length * np.cos(angle)
        y2 = y + length * np.sin(angle)
        # color by depth
        col_t = depth / 11.0
        col = (Forest if col_t < 0.55 else Amber)
        ax.plot([x, x2], [y, y2], color=col, lw=lw, solid_capstyle='round')
        new_lw = max(lw * 0.72, 0.3)
        spread = 0.45 + 0.05 * rng.normal()
        branch(x2, y2, angle + spread, length * 0.74,
               depth - 1, new_lw)
        branch(x2, y2, angle - spread, length * 0.7,
               depth - 1, new_lw)

    branch(0.0, 0.0, np.pi / 2, 1.0, 11, 4.5)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    save_fig('ch_fmh_fractal_tree')


# =============================================================================
# 10. FRACTIONAL BROWNIAN MOTION (4 H values)
# =============================================================================
def fbm_circulant(H, n, seed=0):
    """Davies-Harte / circulant embedding for exact fBm sample."""
    rng = np.random.default_rng(seed)
    # Covariance of fGn
    k = np.arange(n)
    g = 0.5 * (np.abs(k - 1) ** (2 * H)
               - 2 * np.abs(k) ** (2 * H)
               + np.abs(k + 1) ** (2 * H))
    # circulant first row
    r = np.concatenate([g, [0], g[1:][::-1]])
    lam = np.fft.fft(r).real
    lam = np.maximum(lam, 0.0)
    W = (rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n))
    Z = np.fft.fft(np.sqrt(lam) * W) / np.sqrt(2 * n)
    fgn = Z.real[:n]
    fbm = np.cumsum(fgn)
    return fbm


def plot_fbm_paths():
    print("11. fBm SAMPLE PATHS (4 H values)")
    Hs = [0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cols_h = [Crimson, MainBlue, Forest, Purple]
    interp = ['anti-persistent', 'random walk (Brownian)',
              'weakly persistent', 'strongly persistent']

    for ax, H, col, label in zip(axes.ravel(), Hs, cols_h, interp):
        for seed in range(3):
            path = fbm_circulant(H, 1500, seed=seed * 11 + 1)
            ax.plot(path, lw=0.7, color=col, alpha=0.55 + 0.15 * seed)
        ax.set_title(f'H = {H} ({label})', fontweight='bold', fontsize=9)
        ax.set_xlabel('Time step')
        ax.set_ylabel(r'$B_H(t)$')

    plt.tight_layout()
    save_fig('ch_fmh_fbm_paths')


# =============================================================================
# 11. SELF-SIMILARITY of price-like fBm at multiple scales
# =============================================================================
def plot_self_similarity_zoom():
    print("12. SELF-SIMILARITY zoom for fBm with H=0.7")
    np.random.seed(0)
    H = 0.7
    n_total = 8192
    path = fbm_circulant(H, n_total, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    spans = [n_total, n_total // 8, n_total // 64]
    starts = [0, 1500, 1900]

    for ax, span, start in zip(axes, spans, starts):
        seg = path[start:start + span]
        x = np.arange(len(seg))
        ax.plot(x, seg, color=MainBlue, lw=0.8)
        ax.fill_between(x, seg.min(), seg, color=MainBlue, alpha=0.10)
        ax.set_title(f'n = {span} steps', fontweight='bold', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle('Statistical self-similarity: fBm (H=0.7) at different scales',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('ch_fmh_selfsimilarity')


# =============================================================================
# 12. BOX-COUNTING DIMENSION ILLUSTRATION
# =============================================================================
def plot_box_counting():
    print("13. BOX-COUNTING DIMENSION (Koch curve)")
    # Generate Koch curve
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    pts = np.array(koch_curve(A, B, 5))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    grid_sizes = [1/8, 1/16, 1/32, 1/64]
    counts = []

    for ax, eps in zip(axes, grid_sizes):
        ax.plot(pts[:, 0], pts[:, 1], color=MainBlue, lw=0.6)
        # Box count
        ix = np.floor(pts[:, 0] / eps).astype(int)
        iy = np.floor(pts[:, 1] / eps).astype(int)
        boxes = set(zip(ix.tolist(), iy.tolist()))
        N = len(boxes)
        counts.append(N)
        # Draw the occupied boxes
        for (i, j) in boxes:
            rect = plt.Rectangle((i * eps, j * eps), eps, eps,
                                 facecolor=Crimson, alpha=0.18,
                                 edgecolor=Crimson, lw=0.3)
            ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(rf'$\varepsilon = {eps:.4f}$, $N(\varepsilon) = {N}$',
                     fontsize=9, fontweight='bold')

    eps_arr = np.array(grid_sizes)
    N_arr = np.array(counts)
    slope, _ = np.polyfit(np.log(1.0 / eps_arr), np.log(N_arr), 1)
    fig.suptitle(rf'Estimated box-counting fractal dimension: $D \approx {slope:.3f}$  '
                 rf'(Koch theoretical: $\log 4 / \log 3 \approx 1.262$)',
                 fontsize=10, fontweight='bold', y=1.02, color=MainBlue)

    plt.tight_layout()
    save_fig('ch_fmh_boxcount')


# =============================================================================
# 13. EMH vs FMH visual contrast
# =============================================================================
def plot_emh_vs_fmh():
    print("14. EMH vs FMH (panel)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Left column: EMH world (random walk, gaussian)
    np.random.seed(0)
    rw = np.cumsum(np.random.normal(0, 1, 2000))
    ret_rw = np.random.normal(0, 1, 50000)

    # Right column: FMH world (fBm H=0.75 + mixed scales)
    fbm_path = fbm_circulant(0.75, 2000, seed=11) * 1.5
    # heavier tails returns
    np.random.seed(42)
    ret_fmh = np.concatenate([
        np.random.normal(0, 0.8, 47000),
        np.random.standard_t(2.5, 3000) * 1.2,
    ])

    # (0,0) EMH price path
    ax = axes[0, 0]
    ax.plot(rw, color=MainBlue, lw=0.7)
    ax.set_title('EMH: Gaussian random walk ($H=0.5$)',
                 fontweight='bold', fontsize=10, color=MainBlue)
    ax.set_xlabel('Time'); ax.set_ylabel('Price (log)')

    # (0,1) FMH price path
    ax = axes[0, 1]
    ax.plot(fbm_path, color=Crimson, lw=0.7)
    ax.set_title('FMH: fractional Brownian motion ($H=0.75$)',
                 fontweight='bold', fontsize=10, color=Crimson)
    ax.set_xlabel('Time'); ax.set_ylabel('Price (log)')

    # (1,0) EMH return distribution
    ax = axes[1, 0]
    ax.hist(ret_rw, bins=120, density=True, color=MainBlue, alpha=0.6,
            edgecolor='none')
    xg = np.linspace(-6, 6, 400)
    ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi),
            color='black', lw=1.0, label='N(0,1)')
    ax.set_xlim(-6, 6)
    ax.set_title('EMH: Gaussian distribution', fontweight='bold',
                 fontsize=10, color=MainBlue)
    ax.set_xlabel('Standardized return'); ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.20), ncol=1)

    # (1,1) FMH return distribution (fat tails)
    ax = axes[1, 1]
    ret_std = (ret_fmh - ret_fmh.mean()) / ret_fmh.std()
    ax.hist(ret_std, bins=200, density=True, color=Crimson, alpha=0.6,
            edgecolor='none')
    ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi),
            color='black', lw=1.0, linestyle='--', label='N(0,1) (reference)')
    ax.set_yscale('log')
    ax.set_xlim(-6, 6)
    ax.set_ylim(1e-4, 1.0)
    ax.set_title('FMH: heavy tails (log scale)',
                 fontweight='bold', fontsize=10, color=Crimson)
    ax.set_xlabel('Standardized return'); ax.set_ylabel('Density (log)')
    ax.legend(frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.20), ncol=1)

    plt.tight_layout()
    save_fig('ch_fmh_emh_vs_fmh')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SFM FMH CHAPTER: FRACTAL VISUAL GALLERY")
    print(f"Chart directory: {CHART_DIR}")
    print("=" * 70)

    plot_mandelbrot_gallery()
    plot_mandelbrot_single()
    plot_julia_gallery()
    plot_barnsley_fern()
    plot_sierpinski()
    plot_koch_snowflake()
    plot_dragon_curve()
    plot_cantor()
    plot_coastline()
    plot_fractal_tree()
    plot_fbm_paths()
    plot_self_similarity_zoom()
    plot_box_counting()
    plot_emh_vs_fmh()

    print("\nALL FRACTAL FIGURES GENERATED")
