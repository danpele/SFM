"""SFM (Statistica piețelor financiare / Statistics of Financial Markets): web version of a lecture, one scrollable HTML page
per language, generated from the Beamer source. Ported from the MFM course (MFM/latex/build_web_lecture.py); the guide for
writing the course text and checking the pages is latex/WEB_GUIDE.md.

usage: cd 2026/latex && python3 build_web_lecture.py 4            (chapter 4; "all" builds chapters 0-16)
       python3 build_web_lecture.py 4 --hashes                    (prints the key and hash of every frame, for the course text)
       python3 build_web_lecture.py 4 --keys                      (prints the citation keys of the deck)
writes web/ch{N}_ro.html and web/ch{N}.html (+ web/img/, web/slides/ and a log web/ch{N}_log.txt).
The decks are the ones the course site links (assets/course-data.js, 'slides' entries): RO/Cursuri/capitolN_*.tex and
EN/Courses/chapterN_*.tex (EN/Courses also holds older decks with the same chapter number: never picked by file name).

Every frame of the .tex (main body and appendix, in document order) becomes an <article id="s<page>"> under its section.
Text, lists, math (kept as LaTeX for MathJax), links, blocks, columns and simple tables are converted; a chart that has
a block on the interactive page (latex/build_interactive_chN.py, BLOCKS) is replaced by that live block, driven by
interactive/chN.js; other images are shown as PNG. A frame the converter cannot render faithfully (TikZ, unknown macro or
environment, multirow ...) is shown as the image of its PDF page, with a link to that page, and logged with the reason.

Course text (optional): latex/web_text/ch{N}_{ro,en}.md holds prose written for the web page. It is inserted before the frame's
own content, which then sits below in a compact „Pe scurt” / "In brief" box. Format:
  # <section title>                  text up to the first ## is the section introduction
  ## <frame title> {h=<hash>}        text of that frame; the hash is that of the frame's .tex source (--hashes prints them)
  @after                             (optional line) the paragraphs below it go after the live block of the frame
  ## @end                            synthesis paragraph(s) of the current section, after its last frame
  ## @intro                          (optional, before the first #) chapter opener after the title frame: epigraph, motivation,
                                     opening chart, objectives, chapter plan (directives: see intro_html); missing -> no opener
Paragraphs are separated by blank lines; $...$ and $$...$$ are math (MathJax), [text](url) links, **bold**, *italic*.
<!-- ... --> is a comment.
Numbered equations: a display paragraph `$$ ... $$ {#eq:label}` gets the number (N.k), k counted over the labelled displays in
page order; `@eq:label` in the text prints "(N.k)" linked to it. Unlabelled displays stay unnumbered. An unknown or duplicate
label stops the build, and so does a label set or numbering that differs between RO and EN.
Citations: `@Key` prints "Markowitz (1952)", `[@Key]` prints "(Markowitz, 1952)", `[@Key1; @Key2]` several in one bracket;
[[refX]] (the deck's macro) is the same as `@Key`. Keys come from the deck: every \\refX macro whose DOI has an entry in the
deck's bibliography frames, key = the surnames of its label + year ("Ledoit \\& Wolf (2004b)" -> LedoitWolf2004b,
"Pele et al.\\ (2023)" -> PeleEtAl2023). Each citation links to its entry in a list at the end of the page, built from the
keys actually cited, sorted by author, with the entry copied from the deck's bibliography. An unknown key stops the build.
A frame whose hash differs from the current source still gets its text; the build logs "text out of date" in web/chN_log.txt.
On a frame whose live block asks for a prediction (spec.PREDICT), the text after @after and the „Pe scurt” box are veiled by
web.js until the reader presses „Verifică”, runs ▶ to the end or presses „Arată textul”.
Without a text file the page is built exactly as before.
"""
import hashlib
import unicodedata
import html
import importlib
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(ROOT, 'web')
MATHJAX = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js'
PLOTLY = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
SITE = 'https://danpele.github.io/SFM/'
VER = '20261109m'

L = {
 'ro': dict(html_lang='ro', course='Statistica piețelor financiare', chapter='Capitolul', back='Site-ul cursului', pdf_btn='PDF-ul cursului',
            ql='Quantlets', inter='Pagina interactivă', toc='Cuprins', slide='Slide-ul', pdf_slide='Slide {n} în PDF', ask='Întrebare pentru sală',
            answer='Răspuns', site='../index_ro.html#chapters', ticket='Autoevaluare', back_link='Înapoi',
            brief='Pe scurt', synth='Sinteza secțiunii', bib='Bibliografie',
            intro='Introducere', goals='Rezultatele învățării', plan='Planul capitolului', tr='traducere', live='Graficul interactiv',
            thm=dict(defn='Definiție', thm='Teoremă', prop='Propoziție', rmk='Observație')),
 'en': dict(html_lang='en', course='Statistics of Financial Markets', chapter='Chapter', back='Course site', pdf_btn='Lecture PDF',
            ql='Quantlets', inter='Interactive page', toc='Contents', slide='Slide', pdf_slide='Slide {n} in the PDF', ask='Question for the room',
            answer='Answer', site='../index.html#chapters', ticket='Self-Assessment', back_link='Back',
            brief='In brief', synth='Section summary', bib='References',
            intro='Introduction', goals='Learning outcomes', plan='Chapter plan', tr='translation', live='Interactive chart',
            thm=dict(defn='Definition', thm='Theorem', prop='Proposition', rmk='Remark')),
}

COLORS = {'MainBlue': '#1A3A6E', 'AccentBlue': '#1A3A6E', 'IDAred': '#CD0000', 'Crimson': '#DC3545', 'Forest': '#2E7D32', 'Amber': '#B5853F',
          'Orange': '#E67E22', 'Purple': '#8E44AD', 'red': '#CD0000', 'blue': '#1A3A6E', 'DarkGray': '#333333', 'FooterGray': '#646464'}


class Fail(Exception):
    """The frame cannot be rendered faithfully; the reason goes to the log."""


# ------------------------------------------------------------------ low-level scanning
def strip_comments(s):
    out = []
    for line in s.split('\n'):
        i, cut = 0, None
        while i < len(line):
            if line[i] == '\\':
                i += 2
                continue
            if line[i] == '%':
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return '\n'.join(out)


def skip_ws(s, i):
    while i < len(s) and s[i] in ' \t\n':
        i += 1
    return i


def group(s, i):
    """Content of the {...} group starting at s[i] (after optional whitespace) and the index after it."""
    i = skip_ws(s, i)
    if i >= len(s) or s[i] != '{':
        if i < len(s) and s[i] == '\\':                       # single control sequence as argument
            m = re.match(r'\\([a-zA-Z@]+|.)', s[i:])
            return m.group(0), i + m.end()
        if i < len(s):
            return s[i], i + 1
        raise Fail('missing argument')
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise Fail('unbalanced braces')


def opt(s, i):
    """Content of an optional [...] argument at s[i] (after whitespace), or (None, i)."""
    k = skip_ws(s, i)
    if k >= len(s) or s[k] != '[':
        return None, i
    depth, j = 0, k
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c == ']' and depth == 0:
            return s[k + 1:j], j + 1
        j += 1
    raise Fail('unbalanced [')


def env_end(s, i, name):
    """Index of the matching \\end{name} for a \\begin{name} whose content starts at i, and the index after it."""
    depth, pat = 1, re.compile(r'\\(begin|end)\{' + re.escape(name) + r'\}')
    for m in pat.finditer(s, i):
        depth += 1 if m.group(1) == 'begin' else -1
        if depth == 0:
            return m.start(), m.end()
    raise Fail(f'unclosed environment {name}')


def split_top(s, sep_re):
    """Split s at matches of sep_re that are outside braces, environments and math."""
    parts, depth, envd, last, i, math = [], 0, 0, 0, 0, False
    rx = re.compile(sep_re)
    while i < len(s):
        c = s[i]
        if c == '\\':
            m = re.match(r'\\(begin|end)\{[^}]*\}', s[i:])
            if m:
                envd += 1 if m.group(1) == 'begin' else -1
                i += m.end()
                continue
            if depth == 0 and envd == 0 and not math:
                m = rx.match(s, i)
                if m:
                    parts.append(s[last:i])
                    last = i = m.end()
                    continue
            i += 2
            continue
        if c == '$':
            math = not math
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0 and envd == 0 and not math:
            m = rx.match(s, i)
            if m:
                parts.append(s[last:i])
                last = i = m.end()
                continue
        i += 1
    parts.append(s[last:])
    return parts


def parse_macros(src):
    """\\newcommand / \\providecommand / \\renewcommand definitions: name -> (nargs, default of #1 or None, body)."""
    res = {}
    for m in re.finditer(r'\\(?:new|provide|renew)command\*?\s*', src):
        i = m.end()
        try:
            if src[i] == '{':
                name, i = group(src, i)
            else:
                mm = re.match(r'\\[a-zA-Z@]+', src[i:])
                name, i = mm.group(0), i + mm.end()
            n, i = opt(src, i)
            dflt, i = opt(src, i)
            body, i = group(src, i)
        except (Fail, AttributeError, IndexError):
            continue
        res[name.strip().lstrip('\\')] = (int(n) if n else 0, dflt, body)
    return res


def parse_colspec(spec):
    cols, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in 'lcr':
            cols.append(c)
            i += 1
        elif c in 'pmbX':
            i += 1
            if c != 'X':
                _, i = group(spec, i)
            cols.append('l')
        elif c in '><@!':
            _, i = group(spec, i + 1)
        elif c == '*':
            n, i = group(spec, i + 1)
            sub, i = group(spec, i)
            cols += parse_colspec(sub) * int(n)
        else:
            i += 1
    return cols


def esc(t):
    return html.escape(t, quote=False)


def attr(t):
    return html.escape(t, quote=True)


def url_of(u):
    u = u.strip().replace('\\#', '#').replace('\\%', '%').replace('\\_', '_').replace('\\&', '&').replace('\\~', '~')
    return u


def rel_url(u):
    """Course-site URLs become relative to web/ (same deployment); other URLs unchanged."""
    if u.startswith(SITE):
        return '../' + u[len(SITE):]
    # links written relative to the deck (EN/Courses, RO/Cursuri) point to the seminar folder next to it
    for deck_rel, site_rel in (('../Seminars/', '../EN/Seminars/'), ('../Seminarii/', '../RO/Seminarii/')):
        if u.startswith(deck_rel):
            return site_rel + u[len(deck_rel):]
    return u


# ------------------------------------------------------------------ converter
SIZE = {'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge'}
IGNORE0 = SIZE | {'centering', 'raggedright', 'raggedleft', 'noindent', 'nointerlineskip', 'par', 'medskip', 'smallskip', 'bigskip', 'hfill',
                  'vfill', 'null', 'strut', 'bfseries', 'itshape', 'mdseries', 'normalfont', 'relax', 'protect', 'leavevmode', 'unskip',
                  'arraybackslash', 'vss', 'pause', 'ttfamily', 'sffamily', 'rmfamily', 'upshape', 'selectfont', 'displaystyle', 'textstyle', 'justifying', 'sloppy'}
IGNORE1 = {'itemsize', 'vspace', 'color', 'hypersetup', 'addvspace', 'label_', 'pagecolor', 'usetikzlibrary', 'thispagestyle', 'linespread'}
SYM = {'ldots': '…', 'dots': '…', 'textbullet': '•', 'S': '§', '%': '%', '&': '&amp;', '_': '_', '#': '#', '{': '{', '}': '}',
       ',': '\u2009', ';': ' ', ':': ' ', '!': '', ' ': ' ', 'quad': '\u2003', 'qquad': '\u2003\u2003', 'textendash': '–', 'textemdash': '—',
       'textquotedblleft': '“', 'textquotedblright': '”', 'textasciitilde': '~', 'textbackslash': '\\', 'copyright': '©', 'dag': '†',
       'textregistered': '®', 'texttrademark': '™', 'euro': '€', 'textdegree': '°', 'checkmark': '✓', 'ss': 'ß', 'newline': '<br>',
       'linebreak': '<br>', 'enspace': '\u2002', 'thinspace': '\u2009', '/': '', '-': '', 'slash': '/', 'textless': '&lt;', 'textgreater': '&gt;',
       'textbar': '|', 'textquotesingle': '’', 'LaTeX': 'LaTeX', 'TeX': 'TeX', 'aa': 'å', 'o': 'ø', 'l': 'ł', 'i': 'ı', 'O': 'Ø', 'L': 'Ł'}
ACCENT = {"'": '\u0301', '`': '\u0300', '^': '\u0302', '"': '\u0308', '~': '\u0303', 'c': '\u0327', 'v': '\u030c', 'u': '\u0306', '=': '\u0304',
          'H': '\u030b', '.': '\u0307', 'k': '\u0328', 'r': '\u030a'}
EAT_SPACE = {'textquotesingle'}
MATH_ENVS = {'equation', 'equation*', 'align', 'align*', 'gather', 'gather*', 'multline', 'multline*', 'eqnarray', 'eqnarray*', 'displaymath'}


class Conv:
    def __init__(self, deck, lang):
        self.d, self.lang, self.t = deck, lang, L[lang]
        self.macros = deck.macros
        self.depth = 0

    # ---------------------------------------------------------- frame-level state
    def start_frame(self, page, static_live=False):
        self.page, self.live_here, self.ids = page, [], set()
        self.static_live = static_live                       # charts of live blocks shown as static images (hidden opener frame)
        self.ids0 = set(self.d.used_ids)                     # restored if the frame falls back to its slide image

    def fail(self, why):
        raise Fail(why)

    def tid(self, name):
        """HTML id of a hypertarget; emitted once."""
        i = 'h-' + re.sub(r'[^A-Za-z0-9_-]', '_', name)
        if i in self.d.used_ids:
            return ''
        self.d.used_ids.add(i)
        return f'<span class="tgt" id="{i}"></span>'

    # ---------------------------------------------------------- main scanner
    def conv(self, s):
        self.depth += 1
        if self.depth > 60:
            self.fail('macro recursion')
        out, i, n = [], 0, len(s)
        try:
            while i < n:
                c = s[i]
                if c == '\\':
                    h, i = self.control(s, i)
                    out.append(h)
                elif c == '$':
                    h, i = self.inline_math(s, i)
                    out.append(h)
                elif c == '{':
                    g, i = group(s, i)
                    out.append(self.conv(g))
                elif c == '}':
                    i += 1
                elif c == '~':
                    out.append('\u00a0')
                    i += 1
                elif c == '-':
                    if s.startswith('---', i):
                        out.append('—')
                        i += 3
                    elif s.startswith('--', i):
                        out.append('–')
                        i += 2
                    else:
                        out.append('-')
                        i += 1
                elif c == '`':
                    out.append('“' if s.startswith('``', i) else '‘')
                    i += 2 if s.startswith('``', i) else 1
                elif c == "'" and s.startswith("''", i):
                    out.append('”')
                    i += 2
                elif c == "'" and out.count('‘') > out.count('’'):
                    # closes a `...' single quote opened in this group
                    out.append('’')
                    i += 1
                elif c == '&':
                    self.fail('alignment & outside a table')
                elif c in '#^_':
                    self.fail(f'character {c} in text')
                elif c in ' \t\n':
                    j = i
                    while j < n and s[j] in ' \t\n':
                        j += 1
                    out.append(' ')
                    i = j
                else:
                    out.append(esc(c))
                    i += 1
        finally:
            self.depth -= 1
        return ''.join(out)

    def inline_math(self, s, i):
        if s.startswith('$$', i):
            j = s.find('$$', i + 2)
            if j < 0:
                self.fail('unclosed $$')
            return self.display(s[i + 2:j]), j + 2
        j = i + 1
        while j < len(s):
            if s[j] == '\\':
                j += 2
                continue
            if s[j] == '$':
                break
            j += 1
        else:
            self.fail('unclosed $')
        return self.math(s[i + 1:j]), j + 1

    def math(self, m):
        self.check_math(m)
        return '<span class="im">\\(' + esc(m) + '\\)</span>'

    def display(self, m, env=None):
        self.check_math(m)
        body = f'\\begin{{{env}}}{m}\\end{{{env}}}' if env else f'\\[{m}\\]'
        return '<div class="dm">' + esc(body) + '</div>'

    def check_math(self, m):
        for bad in ('\\href', '\\includegraphics', '\\hyperlink', '\\tikz', '\\only', '\\uncover', '\\alt<'):
            if bad in m:
                self.fail(f'{bad} inside math')

    # ---------------------------------------------------------- control sequences
    def control(self, s, i):
        if s.startswith('\\[', i):
            j = s.find('\\]', i + 2)
            if j < 0:
                self.fail('unclosed \\[')
            return self.display(s[i + 2:j]), j + 2
        if s.startswith('\\(', i):
            j = s.find('\\)', i + 2)
            if j < 0:
                self.fail('unclosed \\(')
            return self.math(s[i + 2:j]), j + 2
        if s.startswith('\\\\', i):
            j = i + 2
            if s[j:j + 1] == '*':
                j += 1
            _, j = opt(s, j)
            return '<br>', j
        m = re.match(r'\\([a-zA-Z@]+)(\*?)', s[i:])
        if not m:
            ch = s[i + 1:i + 2]
            if ch in ACCENT:
                return self.accent(s, i + 2, ch)
            if ch == '$':
                return '<span class="usd">$</span>', i + 2
            if ch in SYM:
                return SYM[ch], i + 2
            self.fail(f'control symbol \\{ch}')
        name, star, j = m.group(1), m.group(2), i + m.end()
        if name in ('begin',):
            env, j = group(s, j)
            return self.environment(env.strip(), s, j)
        if name == 'end':
            self.fail('unexpected \\end')
        f = getattr(self, 'c_' + name, None)
        if f:
            return f(s, j)
        if name in IGNORE0:
            return '', j
        if name in IGNORE1:
            _, j = opt(s, j)
            _, j = group(s, j)
            return '', j
        if name in SYM:
            k = j
            if s[k:k + 2] == '{}':
                k += 2
            elif name in EAT_SPACE:                              # TeX drops the space after a control word: Chen\textquotesingle s
                k = skip_ws(s, k)
            return SYM[name], k
        if len(name) == 1 and name in ACCENT:
            return self.accent(s, j, name)
        if name in self.macros:
            return self.expand(name, s, j)
        self.fail(f'macro \\{name}')

    def accent(self, s, j, a):
        arg, j = group(s, j)
        arg = arg.strip()
        import unicodedata
        if arg in ('\\i', 'i') and a != 'c':
            arg = 'i'
        return unicodedata.normalize('NFC', arg + ACCENT[a]), j

    def expand(self, name, s, j):
        n, dflt, body = self.macros[name]
        args = []
        if dflt is not None:
            a, j = opt(s, j)
            args.append(dflt if a is None else a)
        while len(args) < n:
            a, j = group(s, j)
            args.append(a)
        for k, a in enumerate(args, 1):
            body = body.replace('#' + str(k), a)
        return self.conv(body), j

    def expand_text(self, s):
        """Textual expansion of user macros (for URLs)."""
        for _ in range(5):
            m = re.search(r'\\([a-zA-Z@]+)', s)
            if not m or m.group(1) not in self.macros:
                break
            name = m.group(1)
            n, dflt, body = self.macros[name]
            j, args = m.end(), []
            while len(args) < n:
                a, j = group(s, j)
                args.append(a)
            for k, a in enumerate(args, 1):
                body = body.replace('#' + str(k), a)
            s = s[:m.start()] + body + s[j:]
        return s

    def args(self, s, j, n):
        res = []
        for _ in range(n):
            a, j = group(s, j)
            res.append(a)
        return res, j

    # text formatting
    def _wrap(tag, cls=None):
        def f(self, s, j):
            a, j = group(s, j)
            c = f' class="{cls}"' if cls else ''
            return f'<{tag}{c}>{self.conv(a)}</{tag}>', j
        return f
    c_textbf = _wrap('strong')
    c_textit = _wrap('em')
    c_emph = _wrap('em')
    c_textsl = _wrap('em')
    c_texttt = _wrap('code')
    c_underline = _wrap('u')
    c_textsc = _wrap('span', 'sc')
    c_textsf = _wrap('span')
    c_textrm = _wrap('span')
    c_textnormal = _wrap('span')
    c_textup = _wrap('span')
    c_alert = _wrap('span', 'alert')
    c_structure = _wrap('span', 'struct')
    c_mbox = _wrap('span')
    c_fbox = _wrap('span', 'fbox')
    c_intitle = _wrap('span')
    c_beamergotobutton = _wrap('span', 'gbtn')
    c_beamerreturnbutton = _wrap('span', 'gbtn')
    c_footnote = _wrap('span', 'fn')

    def c_textcolor(self, s, j):
        _, j = opt(s, j)
        (col, a), j = self.args(s, j, 2)
        cname = col.split('!')[0].strip()
        st = f' style="color:{COLORS[cname]}"' if cname in COLORS and cname not in ('DarkGray', 'FooterGray') else ''
        return f'<span{st}>{self.conv(a)}</span>', j

    def c_href(self, s, j):
        (u, a), j = self.args(s, j, 2)
        u = rel_url(url_of(self.expand_text(u)))
        return f'<a href="{attr(u)}" target="_blank" rel="noopener">{self.conv(a)}</a>', j

    def c_url(self, s, j):
        u, j = group(s, j)
        u = url_of(u)
        return f'<a href="{attr(u)}" target="_blank" rel="noopener">{esc(u)}</a>', j

    def c_hyperlink(self, s, j):
        (t, a), j = self.args(s, j, 2)
        return f'<a href="#h-{attr(t.strip())}">{self.conv(a)}</a>', j

    def c_hypertarget(self, s, j):
        (t, a), j = self.args(s, j, 2)
        return self.tid(t.strip()) + self.conv(a), j

    def c_label(self, s, j):
        t, j = group(s, j)
        return self.tid(t.strip()), j

    def c_sfmapplink(self, s, j):                               # \sfmapplink{target}{source}{text}
        (tgt, src, a), j = self.args(s, j, 3)
        return f'{self.tid(src.strip())}<a class="jump" href="#h-{attr(tgt.strip())}">{self.conv(a)}</a>', j

    def c_sfmdeeplink(self, s, j):                              # footer button on the slide: moved to the end of the frame
        (tgt, src, a), j = self.args(s, j, 3)
        return f'{self.tid(src.strip())}\x01<a class="jump" href="#h-{attr(tgt.strip())}">{self.conv(a)}</a>\x02', j

    def c_sfmappback(self, s, j):                               # footer button on the slide: moved to the end of the frame
        (src, a), j = self.args(s, j, 2)
        return f'\x01<a class="back" href="#h-{attr(src.strip())}">↩ {self.conv(a)}</a>\x02', j
    c_sfmdeepback = c_sfmappback

    def c_sfmchlink(self, s, j):
        return self.c_href(s, j)
    c_sfmchlinkt = c_sfmchlink

    def c_detokenize(self, s, j):                               # \sfmquantlet: \detokenize{SFM_ch5_garch}, the characters verbatim
        a, j = group(s, j)
        return esc(a.strip()), j

    def c_quantlet(self, s, j):
        _, j = opt(s, j)                                         # \quantlet[i]{name}{url}: second link on the same line
        (name, u), j = self.args(s, j, 2)
        u = url_of(self.expand_text(u))
        return (f'<a class="ql" href="{attr(u)}" target="_blank" rel="noopener"><img src="../logos/ql_logo.png" alt="">'
                f'{self.conv(name)}</a>'), j

    def c_quantinar(self, s, j):
        (name, u), j = self.args(s, j, 2)
        u = url_of(self.expand_text(u))
        return (f'<a class="ql qr" href="{attr(u)}" target="_blank" rel="noopener"><img src="../logos/qr_logo.png" alt="">'
                f'{self.conv(name)}</a>'), j

    def c_interactivlink(self, s, j):
        (u, a), j = self.args(s, j, 2)
        u = url_of(u)
        anchor = u.split('#')[-1] if '#' in u else None
        if anchor and anchor in self.live_here:
            return '', j                                         # the live block is already in this frame
        return f'<a class="ilink" href="{attr(rel_url(u))}">▸ {self.conv(a)}</a>', j
    c_interactivlinkfloat = c_interactivlink

    def c_imgcap(self, s, j):
        a, j = group(s, j)
        return f'<span class="cap">{self.conv(a)}</span>', j

    def c_imgcredit(self, s, j):
        (u, a), j = self.args(s, j, 2)
        return f'<span class="credit"><a href="{attr(url_of(u))}" target="_blank" rel="noopener">{self.conv(a)}</a></span>', j

    def c_titlepage(self, s, j):
        return self.d.title_html(self), j

    # ignored commands with arguments
    def c_setbeamertemplate(self, s, j):
        _, j = group(s, j)
        _, j = opt(s, j)
        _, j = group(s, j)
        return '', j

    def c_setbeamercolor(self, s, j):
        _, j = self.args(s, j, 2)
        return '', j

    def c_usebeamercolor(self, s, j):
        _, j = opt(s, j)
        _, j = group(s, j)
        return '', j

    def c_renewcommand(self, s, j):                             # only table spacing inside a frame
        (name, _), j = self.args(s, j, 2)
        if name.strip() != '\\arraystretch':
            self.fail('macro \\renewcommand')
        return '', j

    def c_setlength(self, s, j):
        _, j = self.args(s, j, 2)
        return '', j
    c_addtolength = c_setlength

    def c_hspace(self, s, j):
        if s[j:j + 1] == '*':
            j += 1
        _, j = group(s, j)
        return ' ', j

    def c_vspace(self, s, j):
        if s[j:j + 1] == '*':
            j += 1
        _, j = group(s, j)
        return '', j

    def c_resizebox(self, s, j):
        (_, _, a), j = self.args(s, j, 3)
        return self.conv(a), j

    def c_scalebox(self, s, j):
        (_, a), j = self.args(s, j, 2)
        return self.conv(a), j

    def c_raisebox(self, s, j):
        _, j = group(s, j)
        _, j = opt(s, j)
        _, j = opt(s, j)
        a, j = group(s, j)
        return self.conv(a), j

    def c_makebox(self, s, j):
        _, j = opt(s, j)
        _, j = opt(s, j)
        a, j = group(s, j)
        return self.conv(a), j

    def c_parbox(self, s, j):
        _, j = opt(s, j)
        (_, a), j = self.args(s, j, 2)
        return self.conv(a), j

    def c_vskip(self, s, j):                                    # \vskip <dimen>
        m = re.match(r'\s*-?\s*[0-9.]*\s*(\\[a-zA-Z]+|[a-z]{2})(\s*(plus|minus)\s*-?[0-9.]*\s*(\\[a-zA-Z]+|[a-z]{2}))*', s[j:])
        return '', j + (m.end() if m else 0)
    c_vspace_ = c_vskip

    def c_ensuremath(self, s, j):
        a, j = group(s, j)
        return self.math(a), j

    def c_shortstack(self, s, j):
        _, j = opt(s, j)
        a, j = group(s, j)
        return self.conv(a), j

    def c_frametitle(self, s, j):
        _, j = group(s, j)
        return '', j

    def c_item(self, s, j):
        self.fail('\\item outside a list')

    def c_includegraphics(self, s, j):
        o, j = opt(s, j)
        f, j = group(s, j)
        return self.figure(f.strip(), o or ''), j

    def c_multirow(self, s, j):
        self.fail('\\multirow')

    # ---------------------------------------------------------- figures
    def figure(self, f, o):
        base = os.path.splitext(os.path.basename(f))[0]
        if base in self.d.blocks:
            anchor = self.d.blocks[base]
            if not (self.static_live or anchor in self.d.live_emitted or anchor in self.live_here):
                self.live_here.append(anchor)
                return f'\x00LIVE:{anchor}\x00'
            src, kind = self.d.image(base, f)                # live block shown earlier: static chart linked to it (no duplicated ids)
            if not src:
                self.fail(f'image {f} not found')
            return (f'<a href="#{anchor}"><img class="fig {kind}" src="{attr(src)}" alt="{attr(self.d.cur_title)}" loading="lazy"></a>'
                    f'<a class="ilink" href="#{anchor}">▸ {self.t["live"]}</a>')
        src, kind = self.d.image(base, f)
        if not src:
            self.fail(f'image {f} not found')
        alt = attr(self.d.cur_title)
        return f'<img class="fig {kind}" src="{attr(src)}" alt="{alt}" loading="lazy">'

    # ---------------------------------------------------------- environments
    def environment(self, env, s, j):
        a, b = env_end(s, j, env)
        body, after = s[j:a], b
        if env in ('itemize', 'enumerate', 'description'):
            return self.lst(env, body), after
        if env in MATH_ENVS:
            return self.display(body, env if env != 'displaymath' else None), after
        if env in ('block', 'exampleblock', 'alertblock'):
            _, k = opt(body, 0)
            title, k = group(body, k)
            cls = {'block': '', 'exampleblock': ' ex', 'alertblock': ' al'}[env]
            t = self.conv(title).strip()
            head = f'<div class="bk-t">{t}</div>' if t else ''
            return f'<div class="bk{cls}">{head}<div class="bk-b">{self.conv(body[k:])}</div></div>', after
        if env == 'columns':
            return self.columns(body), after
        if env == 'column':
            self.fail('column outside columns')
        if env in ('center', 'flushleft', 'flushright'):
            return f'<div class="{env}">{self.conv(body)}</div>', after
        if env in ('minipage', 'cminipage'):
            _, k = opt(body, 0)
            _, k = group(body, k)
            return f'<div class="mp">{self.conv(body[k:])}</div>', after
        if env in ('tabular', 'tabular*', 'tabularx'):
            return self.table(env, body), after
        if env in ('table', 'figure'):
            _, k = opt(body, 0)
            return self.conv(body[k:]), after
        if env in ('defn', 'thm', 'prop', 'rmk', 'theorem', 'definition', 'example', 'proof'):
            name, k = opt(body, 0)
            lab = self.t['thm'].get(env, env.capitalize())
            nm = f' ({self.conv(name)})' if name else ''
            return f'<div class="thm"><strong>{lab}{nm}.</strong> {self.conv(body[k:])}</div>', after
        if env in ('quote', 'quotation'):
            return f'<blockquote>{self.conv(body)}</blockquote>', after
        if env in ('tikzpicture', 'axis', 'overprint', 'lstlisting', 'verbatim'):
            self.fail(f'environment {env}')
        self.fail(f'environment {env}')

    def lst(self, env, body):
        parts = split_top(body, r'\\item(?![a-zA-Z])')
        pre, items = parts[0], parts[1:]
        start = ''
        mc = re.search(r'\\setcounter\s*\{enumi\}\s*\{(\d+)\}', pre)
        if mc and env == 'enumerate':                            # \begin{enumerate}\setcounter{enumi}{k}: list continues at k+1
            start = f' start="{int(mc.group(1)) + 1}"'
            pre = pre[:mc.start()] + pre[mc.end():]
        if self.conv(pre).strip():
            self.fail(f'text before the first \\item in {env}')
        tag = {'itemize': 'ul', 'enumerate': 'ol', 'description': 'dl'}[env]
        out = []
        for it in items:
            lab, k = opt(it, 0)
            h = self.conv(it[k:]).strip()
            if env == 'description':
                out.append(f'<dt>{self.conv(lab or "")}</dt><dd>{h}</dd>')
            else:
                out.append(f'<li>{h}</li>')
        return f'<{tag}{start if tag == "ol" else ""}>' + ''.join(out) + f'</{tag}>'

    def columns(self, body):
        _, k = opt(body, 0)
        cols, i = [], k
        for m in re.finditer(r'\\begin\{column\}', body):
            if m.start() < i:
                continue
            pre = body[i:m.start()]
            if self.conv(pre).strip():
                self.fail('text between columns')
            j = m.end()
            _, j = opt(body, j)
            w, j = group(body, j)
            a, b = env_end(body, j, 'column')
            mw = re.match(r'\s*([0-9.]+)\s*\\(textwidth|linewidth|columnwidth|paperwidth)', w)
            frac = float(mw.group(1)) if mw else 0.5
            cols.append([frac, self.conv(body[j:a])])
            i = b
        if self.conv(body[i:]).strip():
            self.fail('text after the last column')
        if not cols:
            self.fail('columns without column')
        lead = []
        for c in cols:                                           # a live block leaves its column and spans the frame
            for anchor in re.findall(r'\x00LIVE:([^\x00]+)\x00', c[1]):
                lead.append(f'\x00LIVE:{anchor}\x00')
                c[1] = c[1].replace(f'\x00LIVE:{anchor}\x00', '')
        cols = [c for c in cols if re.sub(r'<br>|\s', '', c[1])]
        lead = ''.join(lead)
        if len(cols) == 1:
            return lead + f'<div class="col1">{cols[0][1]}</div>'
        tmpl = ' '.join(f'minmax(0, {round(100 * c[0])}fr)' for c in cols)
        return lead + f'<div class="cols" style="grid-template-columns: {tmpl}">' + ''.join(f'<div class="col">{c[1]}</div>' for c in cols) + '</div>'

    def table(self, env, body):
        k = 0
        if env in ('tabular*', 'tabularx'):
            _, k = group(body, k)
        _, k = opt(body, k)
        spec, k = group(body, k)
        aligns = parse_colspec(spec)
        rows = split_top(body[k:], r'\\\\(\[[^\]]*\])?')
        out_rows, head_end, ncols = [], None, len(aligns)
        rule_re = re.compile(r'\s*\\(toprule|midrule|bottomrule|hline|cmidrule(\([^)]*\))?\{[^}]*\}|addlinespace(\[[^\]]*\])?|cline\{[^}]*\}|rowcolor(\[[^\]]*\])?\{[^}]*\}|noalign\{[^}]*\})\s*')
        for r in rows:
            while True:
                m = rule_re.match(r)
                if not m:
                    break
                if m.group(1) == 'midrule' and head_end is None and out_rows:
                    head_end = len(out_rows)
                r = r[m.end():]
            if not r.strip():
                continue
            cells, html_cells, col = split_top(r, r'&'), [], 0
            for c in cells:
                c = c.strip()
                span, al = 1, aligns[col] if col < len(aligns) else 'l'
                mm = re.match(r'\\multicolumn\s*\{(\d+)\}', c)
                if mm:
                    (n, sp, txt), _ = self.args(c, len('\\multicolumn'), 3)
                    span, al = int(n), (parse_colspec(sp) or ['l'])[0]
                    c = txt
                c = re.sub(r'\\cellcolor(\[[^\]]*\])?\{[^}]*\}', '', c)
                st = {'r': ' class="r"', 'c': ' class="c"'}.get(al, '')
                cs = f' colspan="{span}"' if span > 1 else ''
                html_cells.append((cs, st, self.conv(c).strip()))
                col += span
            if col > max(ncols, 1):
                self.fail('table row wider than its column spec')
            out_rows.append(html_cells)
        if not out_rows:
            self.fail('empty table')
        th = out_rows[:head_end] if head_end else []
        tb = out_rows[head_end:] if head_end else out_rows
        h = '<div class="tbl"><table>'
        if th:
            h += '<thead>' + ''.join('<tr>' + ''.join(f'<th{cs}{st}>{x}</th>' for cs, st, x in r) + '</tr>' for r in th) + '</thead>'
        h += '<tbody>' + ''.join('<tr>' + ''.join(f'<td{cs}{st}>{x}</td>' for cs, st, x in r) + '</tr>' for r in tb) + '</tbody>'
        return h + '</table></div>'


# ------------------------------------------------------------------ deck
class Deck:
    def __init__(self, n, lang, tex, spec):
        self.n, self.lang, self.texp, self.spec = n, lang, tex, spec
        raw = open(os.path.join(ROOT, tex + '.tex'), encoding='utf-8').read()
        self.raw = raw
        s = strip_comments(raw)
        k = s.index('\\begin{document}')
        self.pre, self.body = s[:k], s[k + len('\\begin{document}'):s.index('\\end{document}')]
        common = strip_comments(open(os.path.join(ROOT, 'latex', 'preamble.tex'), encoding='utf-8').read())
        self.macros = parse_macros(common)
        self.macros.update(parse_macros(self.pre))
        self.blocks = {chart: anchor for anchor, chart, _ in spec.BLOCKS} if spec else {}
        self.live_used, self.used_ids = set(), set()
        self.live_emitted = set()                                   # live blocks already placed on the page (each appears once)
        self.cur_title = ''
        self.pdf = os.path.join(ROOT, tex + '.pdf')
        self.viewer = '../pdfjs/web/viewer.html?file=../../' + tex + '.pdf'
        nav = open(os.path.join(ROOT, tex + '.nav'), encoding='utf-8').read()
        self.pages = [(int(a), int(b)) for a, b in re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', nav)]
        self.meta = {}
        for key in ('title', 'subtitle', 'author', 'institute'):
            m = re.search(r'\\' + key + r'\s*(\[[^\]]*\])?\s*\{', self.pre)
            if m:
                self.meta[key], _ = group(self.pre, m.end() - 1)

    def items(self):
        """('section', title) and ('frame', options, title, body) in document order."""
        out = []
        rx = re.compile(r'\\(section|appendix)\b\*?|\\begin\{frame\}')
        i = 0
        while True:
            m = rx.search(self.body, i)
            if not m:
                break
            if m.group(0).startswith('\\section'):
                j = m.end()
                _, j = opt(self.body, j)
                t, j = group(self.body, j)
                out.append(('section', t))
                i = j
            elif m.group(0).startswith('\\appendix'):
                i = m.end()
            else:
                j = m.end()
                o, j = opt(self.body, j)
                if self.body[j:j + 1] == '<':
                    j = self.body.index('>', j) + 1
                    o2, j = opt(self.body, j)
                    o = o or o2
                title = ''
                jj = skip_ws(self.body, j)
                if self.body[jj:jj + 1] == '{':
                    title, j = group(self.body, jj)
                a, b = env_end(self.body, j, 'frame')
                body = self.body[j:a]
                if not title:
                    mt = re.search(r'\\frametitle\s*\{', body)
                    if mt:
                        title, _ = group(body, mt.end() - 1)
                out.append(('frame', o or '', title, body))
                i = b
        return out

    def image(self, base, f):
        """URL (relative to web/) of an image file and its kind."""
        for ext in ('.png', '.jpg', '.jpeg', '.svg'):
            if os.path.exists(os.path.join(ROOT, 'charts', base + ext)):
                return f'../charts/{base}{ext}', 'chart'
        if os.path.exists(os.path.join(ROOT, 'charts', base + '.pdf')):
            import pymupdf
            os.makedirs(os.path.join(OUT, 'img'), exist_ok=True)
            dst = os.path.join(OUT, 'img', base + '.png')
            with pymupdf.open(os.path.join(ROOT, 'charts', base + '.pdf')) as d:
                d[0].get_pixmap(dpi=150).save(dst)
            return f'img/{base}.png', 'chart'
        ext0 = os.path.splitext(f)[1]
        for d, kind in (('photos', 'photo'), ('logos', 'logo')):
            for ext in ([ext0] if ext0 else []) + ['.jpg', '.jpeg', '.png']:
                if os.path.exists(os.path.join(ROOT, d, base + ext)):
                    return f'../{d}/{base}{ext}', kind
        return None, None

    def title_html(self, cv):
        t = cv.t
        sub = cv.conv(self.meta.get('subtitle', '')).strip()
        au = cv.conv(self.meta.get('author', '')).strip()
        inst = cv.conv(self.meta.get('institute', '')).strip()
        return (f'<div class="tp"><p class="tp-sub">{sub}</p><p class="tp-au">{au}</p>'
                f'<p class="tp-in">{inst}</p></div>')

    def live_block(self, anchor, W):
        sp, t = self.spec, L[self.lang]
        ctl, ch = W[anchor]
        q, a = sp.QA[self.lang][anchor]
        return (f'<section class="blk wlive" id="{anchor}"><div class="live"><div class="ctls">{ctl}</div>{ch}'
                f'<ul class="out" id="{sp.PFX[anchor]}-out"></ul></div>'
                f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>'
                f'<details><summary>{t["answer"]}</summary><p>{a}</p></details></section>')


def slide_jpg(deck, page):
    import pymupdf
    os.makedirs(os.path.join(OUT, 'slides'), exist_ok=True)
    name = f'ch{deck.n}_p{page}_{deck.lang}.jpg'
    if not hasattr(deck, '_doc'):
        deck._doc = pymupdf.open(deck.pdf)
    deck._doc[page - 1].get_pixmap(dpi=130).save(os.path.join(OUT, 'slides', name), jpg_quality=82)
    return 'slides/' + name


def chapter_files(n):
    spec = None
    try:
        sys.path.insert(0, os.path.join(ROOT, 'latex'))
        spec = importlib.import_module(f'build_interactive_ch{n}')
    except ImportError:
        pass
    if spec and hasattr(spec, 'TEX'):
        return spec.TEX, spec
    return site_decks()[n], spec


def site_decks():
    """{chapter: {'ro': 'RO/Cursuri/capitolN_...', 'en': 'EN/Courses/chapterN_...'}}: the lecture decks the course site links
    (assets/course-data.js, pdf('slides', ...) entries). EN/Courses holds older decks with the same chapter number, so the
    deck is never chosen by file name."""
    src = open(os.path.join(ROOT, 'assets', 'course-data.js'), encoding='utf-8').read()
    res = {}
    for d, stem, num in re.findall(r"pdf\('slides',\s*'((?:EN/Courses|RO/Cursuri)/((?:chapter|capitol)(\d+)_[^'/]*))\.pdf'", src):
        lang = 'en' if d.startswith('EN/') else 'ro'
        res.setdefault(int(num), {})[lang] = d
    for k, v in res.items():
        if set(v) != {'ro', 'en'}:
            raise SystemExit(f'assets/course-data.js: chapter {k} lacks a RO or EN lecture deck: {v}')
    return res


# ------------------------------------------------------------------ course text (latex/web_text/chN_lang.md)
LINK_MACROS = {'href': 2, 'sfmchlink': 2, 'sfmchlinkt': 2, 'hyperlink': 2, 'sfmapplink': 3, 'sfmdeeplink': 3}


def title_plain(macros, t, depth=0):
    """Plain text of a frame title (key of the course text, heading of a fallback frame, alt texts): a link macro keeps its text
    (\\href{url}{Chapter 1} -> Chapter 1), a citation macro its label (\\refCont -> Cont (2001)), \\textquotesingle is ’."""
    from interactive_common import latex_text
    out, i, rx = [], 0, re.compile(r'\\([a-zA-Z]+)')
    while i < len(t):
        m = rx.match(t, i)
        if not m:
            out.append(t[i])
            i += 1
            continue
        name, j = m.group(1), m.end()
        try:
            if name in LINK_MACROS:
                for _ in range(LINK_MACROS[name] - 1):
                    _, j = group(t, j)
                a, j = group(t, j)
                out.append(title_plain(macros, a, depth + 1))
                i = j
                continue
            if name == 'textquotesingle':
                out.append('’')
                i = j + 2 if t[j:j + 2] == '{}' else skip_ws(t, j)
                continue
            if name.startswith('ref') and name in macros and macros[name][0] == 0 and depth < 5:
                out.append(title_plain(macros, macros[name][2], depth + 1))
                i = j
                continue
        except Fail:
            pass
        out.append(m.group(0))
        i = j
    s = ''.join(out)
    return latex_text(s) if depth == 0 else s


def frame_hash(title, body):
    """Short hash of a frame's .tex source (comments stripped, whitespace normalised)."""
    return hashlib.sha1(re.sub(r'\s+', ' ', title + '\n' + body).strip().encode('utf-8')).hexdigest()[:8]


def load_text(n, lang):
    """{'sec': {section: {'intro': [...], 'end': [...]}}, 'frame': {key: (hash, before, after)}, 'intro': [...] or None} or None (no file)."""
    path = os.path.join(ROOT, 'latex', 'web_text', f'ch{n}_{lang}.md')
    if not os.path.exists(path):
        return None
    src = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(), flags=re.S)
    res, sec, cur = {'sec': {}, 'frame': {}, 'intro': None}, None, None
    for line in src.split('\n'):
        m1, m2 = re.match(r'#\s+(.+?)\s*$', line), re.match(r'##\s+(.+?)\s*$', line)
        if m1 and not line.startswith('##'):
            sec = m1.group(1)
            res['sec'][sec] = {'intro': [], 'end': []}
            cur = res['sec'][sec]['intro']
        elif m2:
            key = m2.group(1)
            if key == '@intro':
                cur = res['intro'] = []
            elif key == '@end':
                cur = res['sec'][sec]['end']
            else:
                mh = re.match(r'(.*?)\s*\{h=([0-9a-f]+)\}$', key)
                title, h = (mh.group(1), mh.group(2)) if mh else (key, '')
                res['frame'][title] = (h, [], [])
                cur = res['frame'][title][1]
        elif line.strip() == '@after' and cur is not res['intro']:
            if cur is not None and res['frame']:
                cur = list(res['frame'].values())[-1][2]
        elif cur is not None:
            cur.append(line)
    return res


class TextError(SystemExit):
    """Error in the course text (unknown citation key or equation label): stops the build."""


def bib_key(label):
    """Citation key of a deck label: 'Ledoit \\& Wolf (2004b)' -> 'LedoitWolf2004b', 'Pele et al.\\ (2023)' -> 'PeleEtAl2023'."""
    m = re.fullmatch(r'(.+?)\s*\((\d{4}[a-z]?)\)', label.strip())
    if not m:
        return None
    names = re.sub(r'et al\.?(\\ )?', ' EtAl ', m.group(1)).replace('\\&', ' ').replace('\\', ' ')
    for x, y in (('{\\o}', 'o'), ('\\o ', 'o'), ('ø', 'o'), ('Ø', 'O'), ('æ', 'ae'), ('ß', 'ss'), ('ł', 'l')):
        names = names.replace(x, y)
    names = ''.join(c for c in unicodedata.normalize('NFKD', names) if not unicodedata.combining(c))
    return ''.join(w[:1].upper() + w[1:] for w in re.findall(r'[A-Za-z]+', names)) + m.group(2)


def deck_bib(deck):
    """{key: dict(macro, names, year, label, doi, entry)}: the deck's \\refX macros whose DOI has an entry (\\item ... (year). \\href{doi}...)
    in the deck's bibliography frames; names/label are the TeX of the citation label, entry the TeX of the bibliography item."""
    items = {}
    for m in re.finditer(r'^[ \t]*\\item\s+(.+?\(\d{4}[a-z]?\)\.\s*\\href\{([^}]*)\}.*?)\s*$', deck.body, re.M):
        items.setdefault(m.group(2).strip(), m.group(1).strip())
    bib = {}
    for name, (_, _, body) in deck.macros.items():
        mm = re.fullmatch(r'\s*\\href\{([^}]*)\}\{(.*)\}\s*', body, re.S)
        if not name.startswith('ref') or not mm or mm.group(1).strip() not in items:
            continue
        label = mm.group(2).strip()
        key = bib_key(label)
        ml = re.fullmatch(r'(.+?)\s*\((\d{4}[a-z]?)\)', label)
        if not key:
            continue
        if key in bib:
            raise TextError(f'{deck.texp}: citation key {key} given by two macros ({bib[key]["macro"]}, {name})')
        bib[key] = dict(macro=name, names=ml.group(1).rstrip('\\ ').strip(), year=ml.group(2), label=label,
                        doi=url_of(mm.group(1).strip()), entry=items[mm.group(1).strip()])   # url_of: \% in a URL is %
    return bib


def cite_html(keys, cv, paren):
    """Author–year citation(s) linked to the entries of the list at the end of the page."""
    ctx = cv.ctx
    out = []
    for k in keys:
        if k not in ctx['bib']:
            raise TextError(f'course text ch{cv.d.n} [{cv.lang}]: unknown citation key "{k}" '
                            f'(known: {", ".join(sorted(ctx["bib"]))})')
        if k not in ctx['used']:
            ctx['used'].append(k)
        b = ctx['bib'][k]
        txt = (cv.conv(b['names']).strip() + ', ' + b['year']) if paren else cv.conv(b['label']).strip()
        out.append(f'<a class="cite" href="#bib-{attr(k)}">{txt}</a>')
    return '(' + '; '.join(out) + ')' if paren else ', '.join(out)


def md_inline(s, cv):
    """Inline markdown of the course text: math, [[refX]] citations, [links](url), **bold**, *italic*."""
    keep = []

    def put(h):
        keep.append(h)
        return f'\x03{len(keep) - 1}\x03'
    s = re.sub(r'\$([^$]+)\$', lambda m: put('<span class="im">\\(' + esc(m.group(1)) + '\\)</span>'), s)
    by_macro = {b['macro']: k for k, b in cv.ctx['bib'].items()}

    def ref_macro(m):
        if m.group(1) not in by_macro:
            raise TextError(f'course text ch{cv.d.n} [{cv.lang}]: [[{m.group(1)}]] is not a deck macro with a bibliography entry')
        return put(cite_html([by_macro[m.group(1)]], cv, False))
    s = re.sub(r'\[\[(ref[A-Za-z]+)\]\]', ref_macro, s)
    s = re.sub(r'(?<![\w@])@eq:([A-Za-z0-9_-]+)', lambda m: put(f'\x05{m.group(1)}\x05'), s)
    s = re.sub(r'\[(@[A-Za-z]\w*(?:\s*;\s*@[A-Za-z]\w*)*)\]',
               lambda m: put(cite_html(re.findall(r'@(\w+)', m.group(1)), cv, True)), s)
    s = re.sub(r'(?<![\w@])@([A-Za-z]\w*)', lambda m: put(cite_html([m.group(1)], cv, False)), s)
    if '@' in s or '[[' in s:
        raise TextError(f'course text ch{cv.d.n} [{cv.lang}]: unresolved citation syntax in: {s[:120]}')
    s = re.sub(r'([(„]?)(\x03\d+\x03)([).,;:”]*)', lambda m: m.group(0) if not (m.group(1) or m.group(3)) else put(
        '<span class="nobr">' + esc(m.group(1)) + keep[int(m.group(2)[1:-1])] + esc(m.group(3)) + '</span>'), s)
    s = esc(s)

    def link(m):
        u = m.group(2)
        ext = '' if u.startswith('#') else ' target="_blank" rel="noopener"'
        return f'<a href="{attr(u)}"{ext}>{m.group(1)}</a>'
    s = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', link, s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])', r'<em>\1</em>', s)
    for _ in range(3):
        s = re.sub(r'\x03(\d+)\x03', lambda m: keep[int(m.group(1))], s)
    return s


def md_html(lines, cv):
    """Paragraphs (blank-line separated) and $$...$$ display math of the course text."""
    out = []
    for par in re.split(r'\n\s*\n', '\n'.join(lines).strip()):
        par = par.strip()
        if not par:
            continue
        me = re.fullmatch(r'\$\$(.+)\$\$\s*\{#eq:([A-Za-z0-9_-]+)\}', par, re.S)
        if me:                                                   # numbered display: the number is set in number_equations (page order)
            out.append(f'<div class="dm eq" id="eq-{me.group(2)}"><div class="eqb">\\[' + esc(me.group(1).strip())
                       + f'\\]</div><span class="eqn">\x06{me.group(2)}\x06</span></div>')
        elif par.startswith('$$') and par.endswith('$$'):
            out.append('<div class="dm">\\[' + esc(par[2:-2].strip()) + '\\]</div>')
        else:
            out.append('<p>' + md_inline(re.sub(r'[ \t\r\n]+', ' ', par), cv) + '</p>')   # keep no-break spaces
    return ''.join(out)


INTRO_DIR = re.compile(r'@(epigraph|portrait|translation|text|chart|goals|plan)\b[ \t]*(.*?)\s*')


def md_list(lines):
    """Items of a markdown list ('- ' lines, continuation lines joined) and the paragraph lines after the list."""
    items, rest, gap = [], [], False
    for line in lines:
        if not rest and re.match(r'\s*-\s+', line):
            items.append(re.sub(r'^\s*-\s+', '', line).strip())
            gap = False
        elif not line.strip():
            gap = True
            if rest:
                rest.append(line)
        elif items and not rest and not gap:
            items[-1] += ' ' + line.strip()
        else:
            rest.append(line)
    return items, rest


def intro_html(lines, cv, deck, items):
    """Chapter opener (## @intro of the course text): epigraph, motivation, opening chart, objectives, chapter plan.
    Directives (each on its own line, in any order; all optional):
      @epigraph <Key> [| <title>]   the quotation below it (verbatim, original language), from the work <Key> of the deck's bibliography;
                                    the title (default: the title of the bibliography entry) links to the DOI
      @portrait <file> | <name> | <credit> | <URL>   portrait of the quoted author left of the epigraph (stacked above it on
                                    narrow screens): <file> relative to the site root (photos/...), <name> as alt text, the
                                    credit line under the portrait, linked to <URL> (the Wikimedia Commons file page)
      @translation                  translation of the quotation (shown below it, marked as translation)
      @text                         paragraphs of the motivation
      @chart <anchor>               opening chart: the static image of the live block <anchor> (BLOCKS), linked to the live block in its
                                    frame, so that no element id of chN.js is duplicated; the lines below are the caption.
                                    Without a live block of that name, <anchor> is the base name of an image in charts/.
      @goals                        '- ' list of what the reader will be able to do; text after the list is a note
      @plan                         '- <section title> | <one-line description>' per section, linked to the section"""
    from interactive_common import latex_text
    t = cv.t
    parts = {}
    cur = None
    for line in lines:
        m = INTRO_DIR.fullmatch(line.strip()) if line.lstrip().startswith('@') else None
        if m:
            cur = m.group(1)
            if cur in parts:
                raise TextError(f'course text ch{deck.n} [{cv.lang}]: @{cur} twice in ## @intro')
            parts[cur] = (m.group(2), [])
        elif cur is None:
            if line.strip():
                raise TextError(f'course text ch{deck.n} [{cv.lang}]: text before the first directive of ## @intro: {line[:80]}')
        else:
            parts[cur][1].append(line)
    out = []
    if 'epigraph' in parts:
        arg, ql = parts['epigraph']
        key, _, title = (x.strip() for x in arg.partition('|'))
        if key not in cv.ctx['bib']:
            raise TextError(f'course text ch{deck.n} [{cv.lang}]: epigraph source "{key}" is not in the deck bibliography')
        b = cv.ctx['bib'][key]
        if not title:
            mt = re.search(r'\\href\{[^}]*\}\{([^}]*)\}', b['entry'])
            title = latex_text(mt.group(1)) if mt else ''
        quote = ' '.join(re.sub(r'^\s*>\s?', '', x).strip() for x in ql if x.strip())
        if not quote:
            raise TextError(f'course text ch{deck.n} [{cv.lang}]: @epigraph without a quotation')
        tr = ''
        if False and 'translation' in parts:                       # no translation under the epigraph (user's choice)
            trq = ' '.join(re.sub(r'^\s*>\s?', '', x).strip() for x in parts['translation'][1] if x.strip())
            if trq:
                tr = f'<p class="epi-tr">„{md_inline(trq, cv)}” <span class="epi-trl">({t["tr"]})</span></p>'
        src = (f'{cite_html([key], cv, False)}, <a class="epi-doi" href="{attr(b["doi"])}" target="_blank" rel="noopener">'
               f'<cite>{esc(title)}</cite></a>')
        ph, cls = '', 'epi'
        if 'portrait' in parts:
            pf = [x.strip() for x in parts['portrait'][0].split('|')]
            if len(pf) != 4 or not all(pf) or not pf[3].startswith('https://'):
                raise TextError(f'course text ch{deck.n} [{cv.lang}]: @portrait needs <file> | <name> | <credit> | <URL>')
            if not os.path.exists(os.path.join(ROOT, pf[0])):
                raise TextError(f'course text ch{deck.n} [{cv.lang}]: @portrait: no file {pf[0]}')
            ph = (f'<div class="epi-ph"><img src="../{attr(pf[0])}" alt="{attr(pf[1])}" width="300" height="400" loading="lazy">'
                  f'<p class="epi-cr"><span class="epi-nm">{esc(pf[1])}</span> '
                  f'<a href="{attr(pf[3])}" target="_blank" rel="noopener">{esc(pf[2])}</a></p></div>')
            cls = 'epi epi-p'
        out.append(f'<figure class="{cls}">{ph}<blockquote lang="en"><p>{md_inline(quote, cv)}</p></blockquote>'
                   f'<figcaption>— {src}</figcaption>{tr}</figure>')
    if 'text' in parts and ''.join(parts['text'][1]).strip():
        out.append(f'<div class="ctext imot">{md_html(parts["text"][1], cv)}</div>')
    if 'chart' in parts:
        anchor, cl = parts['chart']
        base = {a: c for a, c, _ in deck.spec.BLOCKS}.get(anchor) if deck.spec else None
        live = base is not None
        base = base or anchor
        img, _ = deck.image(base, base)
        if not img:
            raise TextError(f'course text ch{deck.n} [{cv.lang}]: @chart {anchor}: no image charts/{base}.png|.pdf')
        cap = md_inline(re.sub(r'\s+', ' ', ' '.join(cl)).strip(), cv)
        lk = ''
        if live:
            ftitle = ''
            for it in items:
                if it[0] == 'frame' and not re.match(r'(Deschidere|Opening)\b', latex_text(it[2])) and re.search(r'\\includegraphics(\[[^\]]*\])?\{[^}]*\b' + re.escape(base) + r'(\.\w+)?\}', it[3]):
                    ftitle = title_plain(deck.macros, it[2])
                    break
            lk = (f' <a class="ilive" href="#{attr(anchor)}">{t["live"]}' + (f': „{esc(ftitle)}”' if cv.lang == 'ro' else f': “{esc(ftitle)}”')
                  + ' →</a>' if ftitle else f' <a class="ilive" href="#{attr(anchor)}">{t["live"]} →</a>')
            img_h = f'<a href="#{attr(anchor)}"><img src="{img}" alt="{attr(re.sub(r"<[^>]+>", "", cap))}" loading="lazy"></a>'
        else:
            img_h = f'<img src="{img}" alt="{attr(re.sub(r"<[^>]+>", "", cap))}" loading="lazy">'
        out.append(f'<figure class="ochart">{img_h}<figcaption>{cap}{lk}</figcaption></figure>')
    if 'goals' in parts:
        li, rest = md_list(parts['goals'][1])
        if li:
            out.append(f'<h3 class="ih">{t["goals"]}</h3><ul class="goals">' + ''.join(f'<li>{md_inline(x, cv)}</li>' for x in li) + '</ul>'
                       + (f'<div class="ctext inote">{md_html(rest, cv)}</div>' if ''.join(rest).strip() else ''))
    if 'plan' in parts:
        secs = [latex_text(x[1]) for x in items if x[0] == 'section']
        li, _ = md_list(parts['plan'][1])
        rows = []
        for x in li:
            name, _, desc = (y.strip() for y in x.partition('|'))
            if name not in secs:
                raise TextError(f'course text ch{deck.n} [{cv.lang}]: @plan: unknown section "{name}" (sections: {"; ".join(secs)})')
            k = secs.index(name) + 1
            rows.append((k, f'<li><a href="#sec-{k}">{esc(name)}</a>' + (f'<span class="plan-d">{md_inline(desc, cv)}</span>' if desc else '') + '</li>'))
        if rows:
            out.append(f'<h3 class="ih">{t["plan"]}</h3><ol class="plan">' + ''.join(r for _, r in sorted(rows)) + '</ol>')
    return f'<section class="wintro" id="intro">{"".join(out)}</section>' if out else ''


def build_page(n, lang, tex, spec, hashes=False):
    t = L[lang]
    deck = Deck(n, lang, tex, spec)
    cv = Conv(deck, lang)
    cv.ctx = dict(bib=deck_bib(deck), used=[])
    text = load_text(n, lang)
    used_text, cur_sec, pending_end = set(), None, ''
    pred = set(getattr(spec, 'PREDICT', {}) or {}) if spec else set()
    seen_titles, seen_old = {}, {}
    W = spec.widgets(lang) if spec else {}
    items = deck.items()
    frames = [x for x in items if x[0] == 'frame']
    assert len(frames) == len(deck.pages), (lang, len(frames), len(deck.pages))
    body, toc, log, sec_n, fi, fnum = [], [], [], 0, 0, 0
    stats = dict(frames=len(frames), converted=0, fallback=0)
    from interactive_common import latex_text
    intro = intro_html(text['intro'], cv, deck, items) if text and text.get('intro') else ''
    if intro:
        toc.append(f'<li><a href="#intro">{t["intro"]}</a></li>')
    in_bib = False                                                # the deck's own bibliography frames: replaced by the generated list
    for it in items:
        if it[0] == 'section':
            if in_bib:
                body.append('</div>')
            in_bib = bool(re.match(r'^\s*(Bibliografie|Bibliography|References)\s*$', latex_text(it[1])))
            if in_bib:
                body.append(pending_end)
                pending_end = ''
                body.append('<div class="deckbib" hidden>')
                continue
            if intro:                                            # chapter opener: after the title frame, before the first section
                body.append(intro)
                intro = ''
            body.append(pending_end)
            pending_end = ''
            sec_n += 1
            title = cv.conv(it[1]).strip()
            body.append(f'<h2 class="sec" id="sec-{sec_n}">{title}</h2>')
            toc.append(f'<li><a href="#sec-{sec_n}">{title}</a></li>')
            cur_sec = latex_text(it[1])
            st = text['sec'].get(cur_sec) if text else None
            if st:
                used_text.add('#' + cur_sec)
                if ''.join(st['intro']).strip():
                    body.append(f'<div class="ctext sintro" id="sec-{sec_n}-intro">{md_html(st["intro"], cv)}</div>')
                if ''.join(st['end']).strip():
                    pending_end = (f'<div class="ctext send" id="sec-{sec_n}-end"><p class="send-t">{t["synth"]}</p>{md_html(st["end"], cv)}</div>')
            continue
        _, o, title, fb = it
        p0, p1 = deck.pages[fi]
        fi += 1
        if 'noframenumbering' not in o:
            fnum += 1
        deck.cur_title = title_plain(deck.macros, title) if title else ''
        opener = bool(text and text.get('intro') and re.match(r'(Deschidere|Opening)\b', deck.cur_title))
        cv.start_frame(p0, static_live=opener)
        flabel = re.search(r'(?:^|,)\s*label\s*=\s*([^,\s]+)', o)    # beamer frame label: target of \hyperlink{label}
        key = deck.cur_title or f'p{p0}'
        seen_titles[key] = seen_titles.get(key, 0) + 1
        if seen_titles[key] > 1:
            key += f' #{seen_titles[key]}'
        okey = (latex_text(title) if title else '') or f'p{p0}'      # key of the earlier title text (links shown as their URL)
        seen_old[okey] = seen_old.get(okey, 0) + 1
        if seen_old[okey] > 1:
            okey += f' #{seen_old[okey]}'
        fh = frame_hash(title, fb)
        if hashes:
            print(f'{lang} p. {p0}: ## {key} {{h={fh}}}')
        ftext = text['frame'].get(key) if text else None
        if ftext:
            used_text.add(key)
        elif text and okey != key and okey in text['frame']:      # course text still under the earlier key: used, and logged
            ftext = text['frame'][okey]
            used_text.add(okey)
            log.append(f'text key to rename: frame p. {p0}: "## {okey}" -> "## {key}"')
        if ftext:
            if ftext[0] != fh:
                log.append(f'text out of date: frame p. {p0} ({key}): stored h={ftext[0] or "-"}, current h={fh}')
        reason = None
        try:
            if 'fragile' in o:
                raise Fail('fragile frame')
            th = cv.conv(title).strip() if title else ''
            h = cv.conv(fb)
        except Fail as e:
            reason = str(e)
        except Exception as e:                                    # parse errors of any kind
            reason = f'parse error: {type(e).__name__}: {e}'
        if reason is not None:                                    # ids taken during the failed conversion are emitted by the fallback
            deck.used_ids = cv.ids0
        ftgt = cv.tid(flabel.group(1)) if flabel else ''
        label = t['pdf_slide'].format(n=fnum) if 'noframenumbering' not in o else f'PDF p. {p0}'
        link = f'<a class="pdfp" href="{deck.viewer}#page={p1}" target="_blank" rel="noopener">{label}</a>'
        if reason is None:
            lives = re.findall(r'\x00LIVE:([^\x00]+)\x00', h)
            deck.live_emitted.update(lives)
            deck.live_used.update(lives)
            if ftext:                                            # live blocks leave the frame content: text, block, „Pe scurt”
                h = re.sub(r'\x00LIVE:[^\x00]+\x00', '', h)
            else:
                for anchor in lives:
                    h = h.replace(f'\x00LIVE:{anchor}\x00', deck.live_block(anchor, W), 1)
            h = re.sub(r'(<img class="fig [^>]*>)\s*<br>', r'\1', h)
            foot = re.findall(r'\x01(.*?)\x02', h)
            h = re.sub(r'\x01.*?\x02', '', h)
            fnav = '<div class="fnav">' + ' '.join(foot) + '</div>' if foot else ''
            stats['converted'] += 1
            head = (f'<div class="fhead">{ftgt}<h3>{th}</h3>{link}</div>' if th else f'<div class="fhead fhead-empty">{ftgt}{link}</div>')
            if ftext:
                stats['text'] = stats.get('text', 0) + 1
                before, after = md_html(ftext[1], cv), md_html(ftext[2], cv)
                live = ''.join(deck.live_block(a, W) for a in lives)
                pa = [a for a in lives if a in pred]
                brief = f'<div class="brief"><p class="brief-t">{t["brief"]}</p><div class="fbody">{h}</div></div>'
                if pa:                                           # web.js veils this group until the prediction is resolved
                    if after:
                        grp = f'<div class="vgroup"><div class="ctext">{after}</div>{brief}</div>'
                        inner = (f'<div class="ctext">{before}</div>' if before else '') + live + grp
                    else:
                        inner = live + f'<div class="vgroup"><div class="ctext">{before}</div>{brief}</div>'
                    data = f' data-pred="{attr(" ".join(pa))}"'
                else:
                    inner = (f'<div class="ctext">{before}</div>' if before else '') + live + (f'<div class="ctext">{after}</div>' if after else '') + brief
                    data = ''
                body.append(f'<article class="frame wtext" id="s{p0}"{data}>{head}{inner}{fnav}</article>')
            else:
                body.append(f'<article class="frame" id="s{p0}">{head}<div class="fbody">{h}{fnav}</div></article>')
        else:
            stats['fallback'] += 1
            log.append(f'p. {p0} ({deck.cur_title}): {reason}')
            th = esc(deck.cur_title)
            img = slide_jpg(deck, p1)
            extra = ''                                            # keep in-page anchors and live blocks of the frame
            for tg in (re.findall(r'\\(?:hypertarget|label)\s*\{([^}]*)\}', fb)
                       + re.findall(r'\\(?:sfmapplink|sfmdeeplink)\s*\{[^}]*\}\s*\{([^}]*)\}', fb)):
                extra += cv.tid(tg.strip())
            for m in re.finditer(r'\\includegraphics(\[[^\]]*\])?\{([^}]*)\}', fb):
                base = os.path.splitext(os.path.basename(m.group(2)))[0]
                if base in deck.blocks and not opener:
                    a = deck.blocks[base]
                    if a in deck.live_emitted:                  # shown live in an earlier frame: link to it
                        extra += f'<a class="ilink" href="#{a}">▸ {t["live"]}</a>'
                    else:
                        deck.live_emitted.add(a)
                        deck.live_used.add(a)
                        extra += deck.live_block(a, W)
            before = md_html(ftext[1], cv) if ftext else ''          # text of the frame above the slide image, @after text below it
            after = md_html(ftext[2], cv) if ftext else ''
            ctx = f'<div class="ctext">{before}</div>' if before else ''
            ctx_after = f'<div class="ctext">{after}</div>' if after else ''
            body.append(f'<article class="frame fb" id="s{p0}"><div class="fhead">{ftgt}<h3>{th}</h3>{link}</div>{ctx}<div class="fbody">'
                        f'<figure class="fbimg"><a href="{deck.viewer}#page={p1}" target="_blank" rel="noopener">'
                        f'<img src="{img}" alt="{attr(t["slide"])} {fnum}: {attr(deck.cur_title)}" loading="lazy"></a>'
                        f'<figcaption><a href="{deck.viewer}#page={p1}" target="_blank" rel="noopener">{label}</a></figcaption></figure>'
                        f'{extra}</div>{ctx_after}</article>')
    body.append(pending_end)
    if in_bib:
        body.append('</div>')
    if text:
        for k in list(text['frame']) + ['#' + k for k in text['sec']]:
            if k not in used_text:
                log.append(f'text without a matching {"section" if k.startswith("#") else "frame"}: {k.lstrip("#")}')
    if spec:
        missing = [a for a, _, _ in spec.BLOCKS if a not in deck.live_emitted]
        if missing:                                               # chN.js needs every block in the DOM
            log.append('live blocks without a frame (hidden): ' + ', '.join(missing))
            body.append('<div hidden>' + ''.join(deck.live_block(a, W) for a in missing) + '</div>')
    body = ''.join(body)
    if text and text.get('intro'):                                # the deck's opener frame duplicates the web opener: keep it hidden (PDF page count unchanged)
        body = re.sub(r'(<article class="frame[^"]*" id="s\d+")([^>]*>(?:(?!</article>).){0,400}?<h3>(?:Deschidere|Opening)\b)',
                      r'\1 hidden\2', body, flags=re.S)
    ids = set(re.findall(r'\sid="([^"]+)"', body))              # a deck's back button whose source link is missing: dropped (logged)
    for m in set(re.findall(r'<a class="back" href="#([^"]+)">.*?</a>', body)):
        if m not in ids:
            log.append(f'back link without its source on the page (removed): #{m}')
            body = re.sub(r'<a class="back" href="#' + re.escape(m) + r'">.*?</a>', '', body)
    body = re.sub(r'<div class="fnav">\s*</div>', '', body)
    body, eqs = number_equations(n, lang, body)
    body = [body]
    if cv.ctx['bib']:                                             # the chapter's bibliography (all entries of the deck), cited ones linked from the text
        def sort_key(k):
            e = cv.ctx['bib'][k]['entry']
            return ''.join(c for c in unicodedata.normalize('NFKD', e) if not unicodedata.combining(c)).casefold()
        li = []
        seen_doi = set()
        for k in sorted(cv.ctx['bib'], key=sort_key):
            if cv.ctx['bib'][k]['doi'] in seen_doi:
                continue
            seen_doi.add(cv.ctx['bib'][k]['doi'])
            b = cv.ctx['bib'][k]
            doi = b['doi']
            li.append(f'<li id="bib-{attr(k)}">{cv.conv(b["entry"]).strip()} '
                      f'<a class="doi" href="{attr(doi)}" target="_blank" rel="noopener">{esc(doi)}</a></li>')
        bib_html = f'<section class="wbib" id="bib"><h2 class="sec">{t["bib"]}</h2><ul class="bib">{"".join(li)}</ul></section>'
    else:
        bib_html = ''
    stats['equations'] = len(eqs)
    stats['citations'] = len(cv.ctx['used'])
    has_ticket = bool(spec and getattr(spec, 'TICKET', None))
    if has_ticket:
        toc.append(f'<li><a href="#check">{t["ticket"]}</a></li>')
    if bib_html:                                                  # last on the page, after the exit ticket (inserted by common.js before .src.foot)
        toc.append(f'<li><a href="#bib">{t["bib"]}</a></li>')
    sub = cv.conv(deck.meta.get('subtitle', '')).strip()
    sub_plain = re.sub(r'<[^>]+>', '', sub)
    other = f'ch{n}.html' if lang == 'ro' else f'ch{n}_ro.html'
    mj_macros = mathjax_macros(deck.macros)
    if spec:                                                      # libraries, then chN.js and web.js in order (no race on window.SFMI)
        from interactive_common import script_loader
        scripts = script_loader([PLOTLY, f'../interactive/ch{n}_data.js?v={VER}', f'../interactive/ch{n}_text_{lang}.js?v={VER}',
                                 f'../interactive/common.js?v={VER}'], f'../interactive/ch{n}.js?v={VER}', [f'web.js?v={VER}'])
    else:
        scripts = f'<script src="web.js?v={VER}" defer></script>'
    inter = (f'<li><a href="../interactive/{spec.L[lang]["file"]}">{t["inter"]}</a></li>' if spec else '')
    ql = getattr(spec, 'QL', f'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_{n:02d}')
    src_note = spec.L[lang].get('src', '') if spec else ''
    page = f'''<!DOCTYPE html>
<html lang="{t["html_lang"]}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(sub_plain)}</title>
<link rel="icon" href="../logos/ida_square.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v=20261016e">
<link rel="stylesheet" href="../interactive/interactive.css?v={VER}">
<link rel="stylesheet" href="web.css?v={VER}">
<script>window.SFM_LANG = '{lang}';
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']], processEscapes: false, macros: {mj_macros} }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'annotation', 'annotation-xml', 'select', 'option', 'button', 'label'] }},
  startup: {{ pageReady: () => MathJax.startup.defaultPageReady().then(() => {{ document.body.setAttribute('data-mj', 'done'); window.dispatchEvent(new Event('sfm-mathjax')); }}) }} }};</script>
<script id="MathJax-script" async src="{MATHJAX}"></script>
{scripts}
</head>
<body class="web">
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a class="home-link" href="{"../index_ro.html" if lang == "ro" else "../index.html"}">{t["course"]}</a></h1><p>{sub}</p></div>
    <a href="https://www.theida.net/" target="_blank" rel="noopener"><img class="header-logo" src="../logos/ida_logo.png" alt="IDA"></a>
  </div>
</header>
<nav><ul>
  <li><a href="{t["site"]}">← {t["back"]}</a></li>
  <li><a href="{deck.viewer}" target="_blank" rel="noopener">{t["pdf_btn"]}</a></li>
  {inter}
  <li><a href="{ql}" target="_blank" rel="noopener">{t["ql"]}</a></li>
  <li class="lang-switcher"><a href="ch{n}.html" class="lang-btn{" active" if lang == "en" else ""}">EN</a><a href="ch{n}_ro.html" class="lang-btn{" active" if lang == "ro" else ""}">RO</a></li>
</ul></nav>
<div class="wlay">
<aside class="wtoc"><details open id="wtoc"><summary>{t["toc"]}</summary><ol>{"".join(toc)}</ol></details></aside>
<main class="wmain">
{"".join(body)}
<p class="src foot">{src_note}</p>
{bib_html}
</main>
</div>
</body>
</html>
'''
    with open(os.path.join(OUT, f'ch{n}_ro.html' if lang == 'ro' else f'ch{n}.html'), 'w', encoding='utf-8') as fh:
        fh.write(page)
    if hasattr(deck, '_doc'):
        deck._doc.close()
    import pymupdf
    with pymupdf.open(deck.pdf) as d:
        stats['pdf_pages'] = d.page_count
    stats['nav_frames'] = len(deck.pages)
    stats['live_blocks'] = len(deck.live_used)
    stats['_eqs'], stats['_cited'] = eqs, sorted(cv.ctx['used'])
    return stats, log


def number_equations(n, lang, h):
    """Numbers (n.k) of the labelled displays in page order (set beside the display, outside its horizontal scroll so that
    they stay visible on narrow screens), and the in-text references (n.k) linked to them."""
    order = re.findall(r'\x06([^\x06]+)\x06', h)
    dup = sorted({x for x in order if order.count(x) > 1})
    if dup:
        raise TextError(f'course text ch{n} [{lang}]: equation label(s) used twice: {", ".join(dup)}')
    num = {lab: f'{n}.{k + 1}' for k, lab in enumerate(order)}
    h = re.sub(r'\x06([^\x06]+)\x06', lambda m: f'({num[m.group(1)]})', h)

    def ref(m):
        if m.group(1) not in num:
            raise TextError(f'course text ch{n} [{lang}]: reference to unknown equation label "{m.group(1)}"')
        return f'<a class="eqref" href="#eq-{m.group(1)}">({num[m.group(1)]})</a>'
    return re.sub(r'\x05([^\x05]+)\x05', ref, h), num


def mathjax_macros(macros):
    """Course macros usable in math (bodies without links, beamer or layout commands), for the MathJax configuration."""
    import json
    bad = re.compile(r'\\(href|hyper|setbeamer|usebeamer|tikz|begin|includegraphics|textcolor|color|scriptsize|tiny|footnotesize|par|vbox|hbox|mbox|beamer|item)')
    res = {}
    for name, (n, dflt, body) in macros.items():
        if not re.fullmatch(r'[A-Za-z]+', name) or bad.search(body) or '@' in body or name.startswith('ref') or 'http' in body or name in ('coursename',):
            continue
        res[name] = [body, n] if n else body
    return json.dumps(res, ensure_ascii=False)


def main(n, hashes=False):
    os.makedirs(OUT, exist_ok=True)
    tex, spec = chapter_files(n)
    report, eqs, cited = [], {}, {}
    for lang in ('ro', 'en'):
        stats, log = build_page(n, lang, tex[lang], spec, hashes)
        eqs[lang], cited[lang] = stats.pop('_eqs'), stats.pop('_cited')
        report.append(f'[{lang}] {stats}')
        report += [('  ' if x.startswith(('text ', 'back link')) else '  fallback ') + x for x in log]
    if eqs['ro'] != eqs['en']:
        raise TextError(f'equation numbering differs between RO and EN: RO {eqs["ro"]} / EN {eqs["en"]}')
    if cited['ro'] != cited['en']:
        raise TextError(f'cited works differ between RO and EN: RO {cited["ro"]} / EN {cited["en"]}')
    if eqs['ro']:
        report.append('equations: ' + ', '.join(f'({v}) {k}' for k, v in eqs['ro'].items()))
    if cited['ro']:
        report.append('cited: ' + ', '.join(cited['ro']))
    with open(os.path.join(OUT, f'ch{n}_log.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(report) + '\n')
    print('\n'.join(report))


def print_keys(n):
    """Citation keys of the chapter's decks (for @Key / [@Key] in the course text)."""
    tex, spec = chapter_files(n)
    for lang in ('ro', 'en'):
        bib = deck_bib(Deck(n, lang, tex[lang], spec))
        print(f'[{lang}] {len(bib)} keys')
        for k, b in sorted(bib.items()):
            print(f'  {k:28s} \\{b["macro"]:22s} {b["doi"]}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    chs = sorted(site_decks()) if args and args[0] == 'all' else [int(a) for a in args] or [4]
    for ch in chs:
        if '--keys' in sys.argv:
            print_keys(ch)
        else:
            if len(chs) > 1:
                print(f'===== chapter {ch}')
            main(ch, '--hashes' in sys.argv)
