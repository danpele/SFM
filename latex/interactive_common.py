"""Shared builder of the interactive chapter pages (SFM/interactive/chN.html and chN_ro.html).

A chapter spec module (latex/build_interactive_chN.py) defines N, TEX, QL, BLOCKS, PFX, SECTIONS, L, widgets(lang),
QA and TEXT_JS, then calls build(module). build writes the two pages, the readout scripts and the slide images.
Each block of BLOCKS is (anchor, chart file, section key); its slide is found from the .tex and the beamer .nav file.
"""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'interactive')
PLOTLY = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'


def script_loader(early, page, rest=()):
    """<script> tags that load the libraries `early` (Plotly, chN_data.js, chN_text_*.js, common.js) in parallel, then, once the
    document is parsed and every library has run, the chapter script `page` (chN.js) and the scripts `rest` (web.js), one after
    the other. The chapter script therefore never starts before window.SFMI exists (a parser-blocking common.js whose request
    fails left it undefined: "Cannot destructure property ... of window.SFMI"). A failed request is sent again, up to 3 times;
    if a library is still missing, the chapter script is skipped (logged in the console) and `rest` still runs."""
    import json
    late = [page] + list(rest)
    pre = ''.join(f'<link rel="preload" as="script" href="{u}">\n' for u in late)
    js = ("(function (early, page, rest) {\n"
          "  'use strict';\n"
          "  let left = early.length, ok = true, parsed = document.readyState !== 'loading', started = false;\n"
          "  function add(src, k, done, ordered) {\n"
          "    const s = document.createElement('script');\n"
          "    s.src = k ? src + (src.indexOf('?') < 0 ? '?' : '&') + 'retry=' + k : src;\n"
          "    s.async = !ordered;\n"
          "    s.onload = () => done(true);\n"
          "    s.onerror = () => { s.remove(); if (k < 3) setTimeout(() => add(src, k + 1, done, ordered), 300 * (k + 1)); else done(false); };\n"
          "    document.head.appendChild(s);\n"
          "  }\n"
          "  function chain(list) { if (list.length) add(list[0], 0, () => chain(list.slice(1)), true); }\n"
          "  function go() {\n"
          "    if (started || left || !parsed) return;\n"
          "    started = true;\n"
          "    if (!ok || !window.SFMI) console.error('SFM: a script of the interactive charts did not load; the live charts are off.');\n"
          "    chain((ok && window.SFMI ? [page] : []).concat(rest));\n"
          "  }\n"
          "  early.forEach(src => add(src, 0, r => { ok = ok && r; left--; go(); }, false));\n"
          "  if (!parsed) document.addEventListener('DOMContentLoaded', () => { parsed = true; go(); });\n"
          "  go();\n"
          f"}})({json.dumps(list(early))}, {json.dumps(page)}, {json.dumps(list(rest))});")
    return pre + '<script>\n' + js + '\n</script>'


def slider(i, label, lo, hi, step, val):
    return f'<div class="ctl"><label for="{i}">{label}: <b id="{i}-v"></b></label><input type="range" id="{i}" min="{lo}" max="{hi}" step="{step}" value="{val}"></div>'


def select(i, label, opts, sel=None):
    o = ''.join(f'<option value="{v}"{" selected" if v == sel else ""}>{t}</option>' for v, t in opts)
    return f'<div class="ctl"><label for="{i}">{label}</label><select id="{i}">{o}</select></div>'


def play(i):
    return f'<button class="btn play" id="{i}-play" type="button"></button>'


def charts(*ids, cls=''):
    return f'<div class="charts {cls}">' + ''.join(f'<div class="chart" id="{i}"></div>' for i in ids) + '</div>'


def balanced(s, i):
    """Content of the brace group that opens at s[i] == '{', and the index after it."""
    depth, j = 0, i
    while True:
        c = s[j]
        if c == '\\':
            j += 2
            continue
        depth += c == '{'
        depth -= c == '}'
        j += 1
        if depth == 0:
            return s[i + 1:j - 1], j


def latex_text(t):
    """Plain text of a frame title: link texts kept, formatting commands and $ removed."""
    GREEK = dict(alpha='α', beta='β', gamma='γ', delta='δ', epsilon='ε', varepsilon='ε', zeta='ζ', eta='η', theta='θ', kappa='κ',
                 lambda_='λ', mu='μ', nu='ν', xi='ξ', pi='π', rho='ρ', sigma='σ', tau='τ', phi='φ', varphi='φ', chi='χ', psi='ψ', omega='ω',
                 Gamma='Γ', Delta='Δ', Theta='Θ', Lambda='Λ', Sigma='Σ', Phi='Φ', Omega='Ω', times='×', le='≤', leq='≤', ge='≥', geq='≥',
                 approx='≈', neq='≠', infty='∞', to='→', cdot='·', pm='±', sqrt='√')
    SUPS = str.maketrans('0123456789+-=()n*', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ*')
    t = re.sub(r'\^\{([0-9+\-=()n*]+)\}|\^([0-9n*])', lambda m: (m.group(1) or m.group(2)).translate(SUPS), t)
    SUBS = str.maketrans('0123456789+-=()aeoxijklmnprstuv', '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₖₗₘₙₚᵣₛₜᵤᵥ')
    t = re.sub(r'_\{([0-9a-z+\-=()]+)\}|_([0-9a-z])', lambda m: (m.group(1) or m.group(2)).translate(SUBS), t)
    t = re.sub(r'\\([a-zA-Z]+)', lambda m: GREEK.get(m.group(1), GREEK.get(m.group(1) + '_', m.group(0))), t)
    t = re.sub(r'\\href\{[^{}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', t)
    for _ in range(3):
        t = re.sub(r'\\(?:textbf|textit|emph|mathrm|mathbf|text|textsc|textsf|texttt|boldsymbol|mbox)\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', t)
    t = t.replace('\\ ', ' ').replace('--', '–').replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_').replace(r'\#', '#').replace('~', ' ')
    t = t.replace('``', '“').replace("''", '”').replace(r'\ldots', '…').replace(r'\dots', '…').replace('$', '')
    t = re.sub(r'\\[a-zA-Z]+\s*', '', t).replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', t).strip()


def frames_of(s):
    """(title, body) of every beamer frame, in order."""
    out = []
    for m in re.finditer(r'\\begin\{frame\}', s):
        i = m.end()
        if s[i:i + 1] == '[':
            i = s.index(']', i) + 1
        title = ''
        if s[i:i + 1] == '{':
            title, i = balanced(s, i)
        out.append((title, s[i:s.index(r'\end{frame}', i)]))
    return out


def slide_pages(sp, lang):
    """Page and title of the slide that shows each chart, from the .tex and the beamer .nav file."""
    tex = os.path.join(ROOT, sp.TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    assert len(frames) == len(pages), (lang, len(frames), len(pages))
    res = {}
    for _, chart, _ in sp.BLOCKS:
        k = next(k for k, (_, body) in enumerate(frames) if re.search(r'includegraphics[^{]*\{' + chart + r'(\.\w+)?\}', body))
        title = re.sub(r'\s*\(\d/\d\)$', '', latex_text(frames[k][0]))
        res[chart] = (int(pages[k][0]), title)
    return res


def page(sp, lang):
    t, W, P = sp.L[lang], sp.widgets(lang), slide_pages(sp, lang)
    viewer = '../pdfjs/web/viewer.html?file=../../' + sp.TEX[lang] + '.pdf'
    body, nav, sec = [], [], None
    for anchor, chart, skey in sp.BLOCKS:
        if skey != sec:
            sec = skey
            body.append(f'<h2 class="sec" id="s-{skey}">{sp.SECTIONS[lang][skey]}</h2>')
            nav.append(f'<a href="#s-{skey}">{sp.SECTIONS[lang][skey]}</a>')
        pg, title = P[chart]
        ctl, ch = W[anchor]
        q, a = sp.QA[lang][anchor]
        body.append(f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{chart}_{lang}.jpg" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    <div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{sp.PFX[anchor]}-out"></ul></div>
  </div>
  <div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>
  <details><summary>{t["answer"]}</summary><p>{a}</p></details>
</section>''')
    return f'''<!DOCTYPE html>
<html lang="{t["html_lang"]}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{t["h1"]}</title>
<link rel="icon" href="../logos/ida_square.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v=20261016e">
<link rel="stylesheet" href="interactive.css?v=20261101a">
<script>window.SFM_LANG = '{lang}';</script>
{script_loader([PLOTLY, f"ch{sp.N}_data.js?v=20261101a", f"ch{sp.N}_text_{lang}.js?v=20261101a", "common.js?v=20261109m"], f"ch{sp.N}.js?v=20261101a")}
</head>
<body>
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a class="home-link" href="{"../index_ro.html" if lang == "ro" else "../index.html"}">{t["course"]}</a></h1><p>{t["h1"]}</p></div>
    <a href="https://www.theida.net/" target="_blank" rel="noopener"><img class="header-logo" src="../logos/ida_logo.png" alt="IDA"></a>
  </div>
</header>
<nav><ul>
  <li><a href="{t["site"]}">← {t["back"]}</a></li>
  <li><a href="{viewer}" target="_blank" rel="noopener">{t["pdf_btn"]}</a></li>
  <li><a href="{sp.QL}" target="_blank" rel="noopener">{t["ql"]}</a></li>
  <li class="lang-switcher"><a href="ch{sp.N}.html" class="lang-btn{" active" if lang == "en" else ""}">EN</a><a href="ch{sp.N}_ro.html" class="lang-btn{" active" if lang == "ro" else ""}">RO</a></li>
</ul></nav>
<main class="imain">
<p class="lead">{t["note"]}</p>
<p class="toc">{" · ".join(nav)}</p>
{"".join(body)}
<p class="src foot">{t["src"]}</p>
</main>
</body>
</html>
'''



def build(sp):
    import pymupdf
    os.makedirs(os.path.join(OUT, 'slides'), exist_ok=True)
    for lang in ('en', 'ro'):
        with open(os.path.join(OUT, sp.L[lang]['file']), 'w', encoding='utf-8') as f:
            f.write(page(sp, lang))
        with open(os.path.join(OUT, f'ch{sp.N}_text_{lang}.js'), 'w', encoding='utf-8') as f:
            f.write(f'// Readouts of the Chapter {sp.N} interactive page, written by latex/build_interactive_ch{sp.N}.py.\n' + sp.TEXT_JS[lang])
        doc = pymupdf.open(os.path.join(ROOT, sp.TEX[lang] + '.pdf'))
        for chart, (pg, _) in slide_pages(sp, lang).items():
            doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(OUT, 'slides', f'{chart}_{lang}.jpg'), jpg_quality=82)
            old = os.path.join(OUT, 'slides', f'{chart}_{lang}.png')
            if os.path.exists(old):
                os.remove(old)
        print('wrote', sp.L[lang]['file'])
