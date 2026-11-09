"""Clickable exercise codes and back buttons in the seminar decks (post-processing, idempotent).

* every exercise code (A1, B2, C1, A2(a) ...) in the text links to the first task frame of that exercise;
* the first task frame of each exercise gets \\hypertarget{ex-ID}{}\\exback{} inline after its first \\item:
  two buttons at the bottom right, "Back"/"Înapoi" (viewer GoBack) and "Exercise list"/"Lista exercițiilor";
* the exercise list (structure slide 2/n, else the exercise map) gets \\hypertarget{ex-list}{}.

Codes after "Seminar N," / "Seminarul N," refer to another seminar and are left as plain text.
Run by Deck.write (latex/sfm_build.py) after the glossary; standalone:
    python3 latex/exercise_links.py [N ...]      # all seminar decks, or only those of chapters N
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ID = r'[ABC]\d{1,2}(?:\([a-e](?:--[a-e])?\))?'
PRE = '\\input{../../latex/preamble}'
LINK_MAC = '\\providecommand{\\exlink}[2]{\\hyperlink{#1}{\\textcolor{MainBlue}{#2}}}'
BACK_MAC = ('\\providecommand{\\exback}{\\begin{tikzpicture}[remember picture,overlay]\\node[anchor=south east] at '
            '([xshift=-3mm,yshift=7mm]current page.south east){\\Acrobatmenu{GoBack}{\\beamerreturnbutton{%s}}\\ '
            '\\hyperlink{ex-list}{\\beamergotobutton{%s}}};\\end{tikzpicture}}')
TITLE = re.compile(r'^\\begin\{frame\}(?:\[[^\]]*\])?\{(' + ID + r')(?=[:\s\]}])')
LIST_TITLES = (re.compile(r"^\\begin\{frame\}(?:\[[^\]]*\])?\{(?:Întrebarea de azi și structura seminarului|"
                          r"Today's Question and Seminar Structure) \(2/\d+\)\}"),
               re.compile(r'^\\begin\{frame\}(?:\[[^\]]*\])?\{(?:Harta exercițiilor|Exercise Map)\}'))
# protected spans: math, URLs and link arguments, existing links/targets, references to other seminars
PROTECT = re.compile(
    r'(\$[^$]*\$'
    r'|\\(?:href|sfmchlink|sfmchlinkt|exlink|hyperlink)\{[^}]*\}\{[^}]*\}'
    r'|\\(?:texttt|url|hypertarget|label|ref|includegraphics|quantlet|sfmquantlet|input|nb|colaburl)\{[^}]*\}'
    r'|\\sfmapplink\{[^}]*\}\{[^}]*\}\{[^}]*\}'
    r'|Seminar(?:ul)?[~ ]\d+,?\s*' + ID + r'(?:(?:,\s*|\s+(?:and|și|or|sau)\s+)' + ID + r')*)')
REF = re.compile(r'(?<![A-Za-z0-9_\\{])(?:(?<=--)|(?<!-))(' + ID + r')(?![0-9A-Za-z_])')   # 'A1--A3' links both


def key(code):
    return 'ex-' + re.sub(r'[^A-Za-z0-9]', '', code)


def frames(lines):
    """(start, end, instructor_only) of every frame; end = index of the \\end{frame} line."""
    out, start = [], None
    for i, l in enumerate(lines):
        if l.startswith('\\begin{frame}'):
            start = i
        elif l.startswith('\\end{frame}') and start is not None:
            prev = next((lines[j].strip() for j in range(start - 1, -1, -1) if lines[j].strip()), '')
            out.append((start, i, prev == '\\ifsolutions'))
            start = None
    return out


def inline_after_item(lines, a, b, text):
    """Put text right after the first '\\item ' between lines a and b; False if there is none."""
    for j in range(a + 1, b):
        k = lines[j].find('\\item ')
        if k >= 0:
            lines[j] = lines[j][:k + 6] + text + lines[j][k + 6:]
            return True
    return False


def process(path):
    s = open(path, encoding='utf-8').read()
    if PRE not in s or '\\begin{document}' not in s:
        return None
    ro = path.endswith('_ro.tex') or '/RO/' in path
    back, lst = ('Înapoi', 'Lista exercițiilor') if ro else ('Back', 'Exercise list')
    if '\\providecommand{\\exlink}' not in s:
        s = s.replace(PRE, PRE + '\n' + BACK_MAC % (back, lst) + '\n' + LINK_MAC, 1)
    lines = s.split('\n')
    body = lines.index('\\begin{document}')
    fr = frames(lines)

    # 1. targets: first student-visible frame of each exercise
    targets, first = {}, {}
    for a, b, instr in fr:
        if a < body or instr:
            continue
        m = TITLE.match(lines[a])
        if m and m.group(1) not in targets:
            targets[m.group(1)] = key(m.group(1))
            first[m.group(1)] = (a, b)
    if not targets:
        return path, 0, 0, 0

    def target_for(code):
        if code in targets:
            return targets[code]
        base = re.match(r'[ABC]\d+', code).group(0)
        if base in targets:
            return targets[base]
        cands = [k for k in targets if k.startswith(base + '(')]
        return targets[cands[0]] if cands else None

    # 2. links in the text (frame titles, comments, graphics lines untouched)
    def rep(m):
        t = target_for(m.group(1))
        return '\\exlink{%s}{%s}' % (t, m.group(1)) if t else m.group(1)

    for i in range(body + 1, len(lines)):
        l = lines[i]
        st = l.lstrip()
        if (st.startswith('\\begin{frame}') or st.startswith('%') or st.startswith('\\frametitle')
                or '\\includegraphics' in l or '\\quantlet' in l):
            continue
        parts = PROTECT.split(l)
        lines[i] = ''.join(p if k % 2 else REF.sub(rep, p) for k, p in enumerate(parts))

    # 3. hypertargets + back buttons, inline after the first \item of the frame
    missing = []
    for code, (a, b) in first.items():
        tag = '\\hypertarget{%s}{}' % targets[code]
        if any(tag in lines[j] for j in range(a, b)):
            continue
        if not inline_after_item(lines, a, b, tag + '\\exback{}'):
            missing.append(code)
    # 4. the exercise list
    if not any('\\hypertarget{ex-list}' in l for l in lines):
        for rx in LIST_TITLES:
            hit = next(((a, b) for a, b, _ in fr if rx.match(lines[a])), None)
            if hit and inline_after_item(lines, hit[0], hit[1], '\\hypertarget{ex-list}{}'):
                break
        else:
            missing.append('ex-list')
    out = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    if missing:
        print('exercise_links: no \\item for', ', '.join(missing), 'in', os.path.relpath(path, ROOT))
    return path, len(targets), out.count('\\exlink{'), out.count('\\exback{}')


def seminar_decks(chapters=None):
    from sfm_chapters import deck
    for n in (chapters if chapters is not None else range(0, 17)):
        for lang in ('en', 'ro'):
            p = os.path.join(ROOT, deck(n, 'seminar', lang))
            if os.path.exists(p):
                yield p


if __name__ == '__main__':
    chs = [int(a) for a in sys.argv[1:]] or None
    for p in seminar_decks(chs):
        r = process(p)
        if r:
            print(os.path.relpath(r[0], ROOT), r[1], 'targets,', r[2], 'links,', r[3], 'back buttons')
