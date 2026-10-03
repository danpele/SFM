#!/usr/bin/env python3
"""
build_variant.py -- assemble an exam variant of "Statistica piețelor financiare / Statistics of Financial Markets"
from the instructor-only problem bank (exam/bank/<lang>/chNN.tex), and compile it with XeLaTeX.

Each bank file holds problems delimited by
    %<problem id=ch05-p2 pts=1>
    ... \\problem[ch05-p2]{Title}{1p} ... \\Q ... \\ifrez\\Rez ... \\barem{...}\\fi
    %</problem>
The RO and EN files use the same ids, so a variant is identical in both languages.

Examples (run from the repository root or from exam/):
    python3 exam/build_variant.py --name V1 --seed 2027 --n 9               # RO, 9 problems from 9 chapters
    python3 exam/build_variant.py --name V1 --seed 2027 --n 9 --lang both   # RO and EN
    python3 exam/build_variant.py --name R1 --ids ch01-p2,ch05-p1,ch10-p4   # chosen problems
    python3 exam/build_variant.py --name V2 --seed 7 --chapters 1-10        # only chapters 1..10
    python3 exam/build_variant.py --bank                                    # whole bank with solutions
    python3 exam/build_variant.py --list                                    # list the problems in the bank

Output: exam/variants/<name>_<lang>.pdf (problems) and <name>_ro_rezolvare.pdf / <name>_en_solutions.pdf
(solutions and grading).
exam/variants/ and exam/bank/ are git-ignored (exam/.gitignore): exam variants and solutions are never committed.
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, 'bank')
OUT = os.path.join(HERE, 'variants')
PAT = re.compile(r'%<problem id=(\S+) pts=([\d.]+)>\s*\n(.*?)%</problem>', re.S)

TXT = {
    'ro': dict(
        exam='EXAMEN', var='Varianta', sol='BAREM ȘI REZOLVARE', bank='BANCA DE PROBLEME (uz intern)',
        head=r'''\noindent Nume și prenume: \dotfill\ \ Grupa: \rule{2.6cm}{0.4pt}\par
\vspace{0.4ex}\noindent Data: \rule{3cm}{0.4pt}\hfill Timp de lucru: 2 ore\par
\vspace{0.6ex}\noindent Toate subiectele sînt obligatorii. Se acordă 1 punct din oficiu. Calculatorul este permis.\par
\vspace{0.3ex}{\small\itshape\noindent La cerințele de interpretare, răspunsul are cel mult 3--5 fraze.
La calcule se punctează atît rezultatul numeric, cît și interpretarea economică sau statistică.
Cînd folosiți o formulă aproximativă, precizați ipotezele principale.\par}
\vspace{0.5ex}{\small\noindent\valoriutilegen\par}''',
        solhead=r'\vspace{0.4ex}\noindent Se acordă 1 punct din oficiu.\par',
        total=lambda p, q: f'Punctaj: {p} puncte + 1 punct din oficiu = {q} puncte.',
        chapter='Capitolul'),
    'en': dict(
        exam='EXAM', var='Variant', sol='SOLUTIONS AND GRADING', bank='PROBLEM BANK (instructor only)',
        head=r'''\noindent Name: \dotfill\ \ Group: \rule{2.6cm}{0.4pt}\par
\vspace{0.4ex}\noindent Date: \rule{3cm}{0.4pt}\hfill Time: 2 hours\par
\vspace{0.6ex}\noindent All problems are compulsory. 1 point is granted by default. A calculator is allowed.\par
\vspace{0.3ex}{\small\itshape\noindent Interpretation answers should be at most 3--5 sentences.
Computations are graded on both the numerical result and its economic or statistical interpretation.
When you use an approximate formula, state its main assumptions.\par}
\vspace{0.5ex}{\small\noindent\valoriutilegen\par}''',
        solhead=r'\vspace{0.4ex}\noindent 1 point is granted by default.\par',
        total=lambda p, q: f'Total: {p} points + 1 point by default = {q} points.',
        chapter='Chapter'),
}


def fmt_pts(p, lang):
    s = f'{p:g}'
    return s.replace('.', ',') if lang == 'ro' else s


def load_bank(lang):
    """Return {id: (chapter, pts, body)} in chapter order, plus the chapter titles."""
    probs, titles = {}, {}
    d = os.path.join(BANK, lang)
    if not os.path.isdir(d):
        sys.exit(f'no bank folder {d}')
    for f in sorted(os.listdir(d)):
        m = re.match(r'ch(\d\d)\.tex$', f)
        if not m:
            continue
        ch = int(m.group(1))
        s = open(os.path.join(d, f), encoding='utf-8').read()
        t = re.search(r'\\banktitle\{\d+\}\{(.*?)\}\s*$', s, re.M)
        titles[ch] = t.group(1) if t else ''
        for pid, pts, body in PAT.findall(s):
            probs[pid] = (ch, float(pts), body.strip() + '\n')
    return probs, titles


def parse_chapters(spec):
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def select(probs, n, seed, chapters):
    """n problems from n different chapters (seeded); if n exceeds the chapters, a chapter is used again."""
    rng = random.Random(seed)
    bych = {}
    for pid, (ch, _, _) in probs.items():
        if ch in chapters:
            bych.setdefault(ch, []).append(pid)
    chs = sorted(bych)
    if not chs:
        sys.exit('no problems in the requested chapters')
    picked = []
    while len(picked) < n:
        pool = [c for c in chs if any(p not in picked for p in bych[c])]
        if not pool:
            break
        k = min(n - len(picked), len(pool))
        for c in sorted(rng.sample(pool, k)):
            picked.append(rng.choice([p for p in bych[c] if p not in picked]))
    return sorted(picked, key=lambda p: (probs[p][0], p))


def document(lang, body, rez, year):
    return ('\\documentclass[10.5pt]{article}\n\\def\\examlang{%s}\n\\def\\headyear{%s}\n'
            '\\input{../sfm_exam_preamble}\n\\rez%s\n\\begin{document}\n' % (lang, year, 'true' if rez else 'false')
            + body + '\n\\end{document}\n')


def compile_tex(path):
    """Compile with XeLaTeX twice. The build files go to a temporary folder (the course folder may refuse writes
    from TeX); the PDF is copied back next to the .tex. Returns True when there are no errors and no overfull boxes."""
    path = os.path.abspath(path)
    d, f = os.path.split(path)
    tmp = tempfile.mkdtemp(prefix='sfm_exam_')
    ok = True
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={tmp}', f],
                           cwd=d, capture_output=True, text=True)
        ok = r.returncode == 0
    stem = f[:-4]
    logf = os.path.join(tmp, stem + '.log')
    log = open(logf, encoding='utf-8', errors='ignore').read() if os.path.exists(logf) else ''
    errs = log.count('\n! ')
    over = len(re.findall(r'Overfull \\[hv]box', log))
    if os.path.exists(os.path.join(tmp, stem + '.pdf')):
        shutil.copyfile(os.path.join(tmp, stem + '.pdf'), path[:-4] + '.pdf')
    if not (ok and errs == 0 and over == 0):
        shutil.copyfile(logf, path[:-4] + '.log') if os.path.exists(logf) else None
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'   {stem}.pdf  errors={errs}  overfull={over}' + ('' if ok else '  (FAILED)'))
    return ok and errs == 0 and over == 0


def build_variant(name, ids, lang, probs, year, date):
    T = TXT[lang]
    total = sum(probs[p][1] for p in ids)
    blocks = '\n'.join(probs[p][2] for p in ids)
    end = rf'\vfill\begin{{center}}\bfseries {T["total"](fmt_pts(total, lang), fmt_pts(total + 1, lang))}\end{{center}}'
    vt = f'{T["exam"]} {date} --- {T["var"]} {name}'.strip()
    os.makedirs(OUT, exist_ok=True)
    out = []
    for rez in (False, True):
        head = (rf'\titlublock{{{vt} --- {T["sol"]}}}' + '\n' + T['solhead']) if rez else \
            (rf'\titlublock{{{vt}}}' + '\n' + T['head'])
        tex = document(lang, head + '\n' + blocks + '\n' + end, rez, year)
        fn = os.path.join(OUT, f'{name}_{lang}' + (('_rezolvare' if lang == 'ro' else '_solutions') if rez else '') + '.tex')
        open(fn, 'w', encoding='utf-8').write(tex)
        out.append(compile_tex(fn))
    return all(out)


def build_bank(lang, probs, titles, year):
    T = TXT[lang]
    parts = [rf'\titlublock{{{T["bank"]}}}']
    cur = None
    for pid in sorted(probs, key=lambda p: (probs[p][0], p)):
        ch = probs[pid][0]
        if ch != cur:
            cur = ch
            parts.append(rf'\clearpage\setcounter{{subj}}{{0}}{{\Large\bfseries\color{{spfblue}} {T["chapter"]} {ch}. '
                         rf'{titles.get(ch, "")}\par}}\vspace{{1ex}}')
        parts.append(probs[pid][2])
    tex = document(lang, '\n'.join(parts), True, year)
    fn = os.path.join(BANK, f'bank_{lang}.tex')
    open(fn, 'w', encoding='utf-8').write(tex)
    return compile_tex(fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='V1', help='variant name, used in the file names and the title')
    ap.add_argument('--seed', type=int, default=None, help='random seed (the same seed gives the same variant)')
    ap.add_argument('--n', type=int, default=9, help='number of problems (default 9: 9 points + 1 by default)')
    ap.add_argument('--chapters', default='1-15', help='chapters to draw from, e.g. 1-10 or 1,2,5-9')
    ap.add_argument('--ids', default='', help='comma-separated problem ids instead of a random draw')
    ap.add_argument('--lang', default='ro', choices=['ro', 'en', 'both'])
    ap.add_argument('--year', default='2027')
    ap.add_argument('--date', default='', help='exam date printed in the title, e.g. "iunie 2027"')
    ap.add_argument('--bank', action='store_true', help='compile the whole bank (with solutions) instead')
    ap.add_argument('--list', action='store_true', help='list the problems in the bank')
    a = ap.parse_args()
    langs = ['ro', 'en'] if a.lang == 'both' else [a.lang]

    if a.list:
        probs, titles = load_bank('ro')
        for pid, (ch, pts, body) in sorted(probs.items(), key=lambda kv: (kv[1][0], kv[0])):
            t = re.search(r'\\problem\[[^\]]*\]\{(.*?)\}\{', body)
            print(f'{pid:10s} {pts:g}p  {t.group(1) if t else ""}')
        print(f'{len(probs)} problems')
        return
    ok = True
    if a.bank:
        for lang in langs:
            probs, titles = load_bank(lang)
            ok &= build_bank(lang, probs, titles, a.year)
        sys.exit(0 if ok else 1)

    probs_ro, _ = load_bank('ro')
    if a.ids:
        ids = [x.strip() for x in a.ids.split(',') if x.strip()]
        bad = [x for x in ids if x not in probs_ro]
        if bad:
            sys.exit(f'unknown ids: {bad}')
        ids = sorted(ids, key=lambda p: (probs_ro[p][0], p))
    else:
        seed = a.seed if a.seed is not None else random.SystemRandom().randrange(10 ** 6)
        ids = select(probs_ro, a.n, seed, parse_chapters(a.chapters))
        print(f'seed={seed}')
    print('problems:', ', '.join(ids))
    for lang in langs:
        probs, _ = load_bank(lang)
        missing = [p for p in ids if p not in probs]
        if missing:
            print(f'   {lang}: missing {missing}, skipped')
            ok = False
            continue
        ok &= build_variant(a.name, ids, lang, probs, a.year, a.date)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
