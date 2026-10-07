"""
sfm_style.py -- chart style of the SFM course (the same as MFM)
===============================================================
  * transparent background (figure, axes, saved files), no grid, no top/right spines;
  * the legend always OUTSIDE the plot, at the bottom centre (legend_outside_bottom), without a frame;
  * colours from the course palette (the LaTeX colours of latex/preamble.tex); no grey series and no grey text:
    text and axes are dark (DarkText), reference lines use a palette colour (dashed);
  * charts saved as PDF (for the slides) and PNG (for the notebooks and the site) in charts/.

Use:
    import sfm_style as st
    st.apply()                                   # once, before the first chart
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, y, color=st.COL['sp500'], label='S&P 500')
    st.legend_outside_bottom(ax, ncol=3)
    st.save_fig('sfm_ch1_returns')               # charts/sfm_ch1_returns.pdf + .png
    st.check_no_grey(fig)                        # optional: raises if a series or a text is grey

Statistics of Financial Markets - Daniel Traian PELE
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# Palette (RGB values of latex/preamble.tex)
MainBlue = '#1A3A6E'
IDAred = '#CD0000'
Forest = '#2E7D32'
Amber = '#B5853F'
Orange = '#E67E22'
Purple = '#8E44AD'
Teal = '#17A2B8'
Crimson = '#DC3545'
DarkText = '#1F2A44'          # text, axes and ticks (dark navy, not grey)

PALETTE = [MainBlue, IDAred, Forest, Amber, Purple, Orange, Teal, Crimson]
# fixed colours for the series used in several chapters
COL = {'sp500': MainBlue, 'bet': IDAred, 'bettr': Orange, 'dax': Teal, 'btc': Amber, 'eth': Crimson,
       'eurron': Forest, 'gold': Purple, 'vix': Crimson, 'stoxx50': Forest, 'ndx': Teal}

_HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(_HERE, '..', '..', 'charts')


def apply():
    """Set the course style for matplotlib."""
    rc = plt.rcParams
    rc['figure.facecolor'] = 'none'
    rc['axes.facecolor'] = 'none'
    rc['savefig.facecolor'] = 'none'
    rc['savefig.transparent'] = True
    rc['axes.grid'] = False
    rc['font.family'] = 'sans-serif'
    rc['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    # font sizes for slides (charts are 9-11 inches wide and shown at 8-14 cm)
    rc['font.size'] = 12
    rc['axes.labelsize'] = 13
    rc['axes.titlesize'] = 13
    rc['xtick.labelsize'] = 11.5
    rc['ytick.labelsize'] = 11.5
    rc['legend.fontsize'] = 11
    rc['axes.spines.top'] = False
    rc['axes.spines.right'] = False
    rc['axes.linewidth'] = 0.6
    rc['lines.linewidth'] = 1.4
    rc['legend.facecolor'] = 'none'
    rc['legend.framealpha'] = 0
    rc['legend.frameon'] = False
    for k in ('text.color', 'axes.labelcolor', 'axes.edgecolor', 'xtick.color', 'ytick.color', 'axes.titlecolor'):
        rc[k] = DarkText
    rc['axes.prop_cycle'] = mpl.cycler(color=PALETTE)


def legend_outside_bottom(ax, ncol=2, y=-0.22, **kw):
    """Place the legend outside the plot, bottom centre (for a figure with several axes, pass the last one
    or use fig.legend with the same arguments)."""
    return ax.legend(loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, **kw)


def fig_legend_bottom(fig, handles=None, labels=None, ncol=3, y=-0.02):
    """One legend for a whole figure (several panels), below the panels."""
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels and not l.startswith('_'):
                    handles.append(h)
                    labels.append(l)
    return fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


# Legibility of multi-panel charts on the slides. A chart gets at most a box of about 0.97 \textwidth by
# 0.55 \textheight on a slide (5.5 x 1.65 inches); its tick labels should then measure at least MIN_PT points.
# A multi-panel figure whose tick labels would be smaller is shrunk (all fonts, lines and markers keep their
# point sizes, so they grow relative to the panels) and laid out again; the data are not touched.
SLIDE_BOX_IN = (5.5, 1.65)
MIN_PT = 6.0
MAX_SHRINK = 1.8
MIN_WIDTH_IN = 3.5


def _panels(fig):
    """Distinct data panels: visible axes with their axis on, without colorbars, twins counted once."""
    boxes = []
    for a in fig.axes:
        if not a.get_visible() or not a.axison or a.get_label() == '<colorbar>':
            continue
        b = tuple(round(x, 3) for x in a.get_position().bounds)
        if b not in boxes:
            boxes.append(b)

    def inside(p, q):
        return p != q and p[0] >= q[0] - 1e-3 and p[1] >= q[1] - 1e-3 and \
            p[0] + p[2] <= q[0] + q[2] + 1e-3 and p[1] + p[3] <= q[1] + q[3] + 1e-3
    return [p for p in boxes if not any(inside(p, q) for q in boxes)]


def _tick_size(fig):
    sizes = [t.get_fontsize() for a in fig.axes if a.get_visible() and a.axison and a.get_label() != '<colorbar>'
             for t in a.get_xticklabels() + a.get_yticklabels() if t.get_text()]
    if not sizes:
        sizes = [mpl.rcParams['xtick.labelsize'] if isinstance(mpl.rcParams['xtick.labelsize'], (int, float))
                 else mpl.rcParams['font.size']]
    sizes.sort()
    return sizes[len(sizes) // 2]


def _below_axes(fig, ax, renderer, leg):
    """True if the legend `leg` of `ax` sits under the axes (a legend outside, at the bottom)."""
    lb, ab = leg.get_window_extent(renderer), ax.get_window_extent(renderer)
    return lb.y1 <= ab.y0 + 0.25 * ab.height and lb.y0 < ab.y0


def _reanchor_legends(fig, pad_pt=3.0):
    """Put every legend that sits below its axes (or below the panels, for a figure legend) just under the tick
    labels and axis labels, which may have grown relative to the panels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible() or not _below_axes(fig, ax, r, leg):
            continue
        tb = ax.get_tightbbox(r, bbox_extra_artists=[])
        ab = ax.get_window_extent(r)
        y = (tb.y0 - ab.y0 - pad_pt * fig.dpi / 72) / ab.height
        leg.set_loc('upper center')
        leg.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)
    if fig.legends:
        bottoms = [a.get_tightbbox(r, bbox_extra_artists=[a.get_legend()] if a.get_legend() else []).y0
                   for a in fig.axes if a.get_visible()]
        fb = fig.bbox
        for leg in fig.legends:
            if leg.get_window_extent(r).y1 > fb.y0 + 0.5 * fb.height:      # a legend at the top stays there
                continue
            y = (min(bottoms) - fb.y0 - pad_pt * fig.dpi / 72) / fb.height
            leg.set_loc('upper center')
            leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)


def _break(txt):
    """Break the longest line of `txt` in two, at ': ', ', ', ' (' or a space near its middle."""
    lines = txt.split('\n')
    k = max(range(len(lines)), key=lambda n: len(lines[n]))
    t = lines[k]
    cuts = [i for i in range(1, len(t) - 1) if t[i] == ' ' and t[i - 1] in ':,;']
    cuts = cuts or [i for i in range(1, len(t) - 1) if t[i] == ' ' and t[i + 1] == '(']
    cuts = cuts or [i for i in range(1, len(t) - 1) if t[i] == ' ']
    if not cuts:
        return None
    i = min(cuts, key=lambda c: abs(c - len(t) / 2))
    lines[k:k + 1] = [t[:i], t[i + 1:]]
    return '\n'.join(lines)


def _wrap_titles(fig):
    """Break panel titles and x-axis labels wider than their panel, and y-axis labels taller than it."""
    changed = False
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        again = False
        for ax in fig.axes:
            ab = ax.get_window_extent(r)
            for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label):
                if t.get_text() and t.get_window_extent(r).width > 1.02 * ab.width and t.get_text().count('\n') < 2:
                    b = _break(t.get_text())
                    if b:
                        t.set_text(b)
                        changed = again = True
            yl = ax.yaxis.label
            if yl.get_text() and yl.get_window_extent(r).height > 1.02 * ab.height and yl.get_text().count('\n') < 2:
                b = _break(yl.get_text())
                if b:
                    yl.set_text(b)
                    changed = again = True
        if not again:
            break
    return changed


def _unclash_xticklabels(fig):
    """Tick labels that now overlap: x labels are turned (45 degrees, or 90 if already steep); y labels are
    thinned (fewer ticks)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    changed = False
    for ax in fig.axes:
        labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        ext = sorted((t.get_window_extent(r) for t in labs), key=lambda b: b.x0)
        if any(a.x1 > b.x0 + 1 for a, b in zip(ext, ext[1:])):
            rot = max(t.get_rotation() % 180 for t in labs)
            if rot < 45:
                for t in labs:
                    t.set_rotation(45)
                    t.set_ha('right')
                    t.set_rotation_mode('anchor')
                changed = True
            elif rot < 89:
                for t in labs:
                    t.set_rotation(90)
                    t.set_ha('center')
                    t.set_rotation_mode('default')
                changed = True
        # y tick labels that touch each other (short panels): fewer ticks on a linear axis
        ylabs = [t for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        yext = sorted((t.get_window_extent(r) for t in ylabs), key=lambda b: b.y0)
        if ax.get_yscale() == 'linear' and any(a.y1 > b.y0 + 0.5 for a, b in zip(yext, yext[1:])):
            n = max(2, len(ylabs) // 2)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=n))
            changed = True
    return changed


def _split_overlapping_legends(fig):
    """Legends of neighbouring panels that now overlap side by side get half as many columns."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legs = [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None and ax.get_legend().get_visible()]
    ext = [lg.get_window_extent(r) for _, lg in legs]
    bad = set()
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = ext[i], ext[j]
            if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                bad.update((i, j))
    for i in bad:
        ax, lg = legs[i]
        ncol = max(1, -(-lg._ncols // 2))
        if ncol == lg._ncols:
            continue
        texts = lg.get_texts()
        ax.legend(lg.legend_handles, [t.get_text() for t in texts], ncol=ncol, loc='upper center',
                  bbox_to_anchor=(0.5, -0.2), bbox_transform=ax.transAxes, frameon=False,
                  fontsize=texts[0].get_fontsize() if texts else None, markerscale=lg.markerscale,
                  handlelength=lg.handlelength, columnspacing=lg.columnspacing)
    return bool(bad)


def legible(fig):
    """Shrink a multi-panel figure so that its tick labels measure at least MIN_PT on the slide."""
    if len(_panels(fig)) < 2:
        return 1.0
    w, h = fig.get_size_inches()
    eff = _tick_size(fig) * min(SLIDE_BOX_IN[0] / w, SLIDE_BOX_IN[1] / h)
    if eff >= MIN_PT - 0.05:
        return 1.0
    k = min(MIN_PT / eff, MAX_SHRINK, max(1.0, w / MIN_WIDTH_IN))   # never narrower than MIN_WIDTH_IN
    fig.set_size_inches(w / k, h / k)
    engine = fig.get_layout_engine()
    constrained = engine is not None and type(engine).__name__ == 'ConstrainedLayoutEngine'
    _wrap_titles(fig)

    def relayout():
        for _ in range(2):
            _reanchor_legends(fig)
            if not constrained:
                try:
                    fig.tight_layout()
                except Exception:                # axes that tight_layout cannot handle: keep the layout
                    pass
        _reanchor_legends(fig)
    relayout()
    for _ in range(3):                           # legends side by side that overlap, tick labels that clash
        a = _split_overlapping_legends(fig)
        b = _unclash_xticklabels(fig)
        if not (a or b):
            break
        relayout()
    return k


def save_fig(name, out_dir=None, show=False):
    """Save the current figure as transparent PDF and PNG (charts/ by default); multi-panel figures are first
    made legible at slide size (legible)."""
    d = out_dir or CHART_DIR
    os.makedirs(d, exist_ok=True)
    legible(plt.gcf())
    plt.savefig(os.path.join(d, f'{name}.pdf'), bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(d, f'{name}.png'), bbox_inches='tight', transparent=True, dpi=180)
    if show:
        plt.show()
    plt.close()
    print(f'   saved {name}')


def _is_grey(c, tol=0.06):
    try:
        r, g, b = to_rgb(c)
    except ValueError:
        return False
    return max(r, g, b) - min(r, g, b) < tol and 0.25 < (r + g + b) / 3 < 0.95


def check_no_grey(fig):
    """House rule: no grey series and no grey text. Raises ValueError listing the offending elements."""
    bad = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            if not ln.get_label().startswith('_') and _is_grey(ln.get_color()):
                bad.append(f'line {ln.get_label()!r}')
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if len(fc) and coll.get_label() and not coll.get_label().startswith('_') and _is_grey(fc[0][:3]):
                bad.append(f'series {coll.get_label()!r}')
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text() and _is_grey(t.get_color()):
                bad.append(f'text {t.get_text()[:30]!r}')
    if bad:
        raise ValueError('grey elements: ' + ', '.join(bad))
