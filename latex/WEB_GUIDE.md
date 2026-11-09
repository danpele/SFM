# Web lectures (SFM) — how to write the course text of chapter N and build the page

Every SFM chapter 0–16 has a web lecture, one scrollable page per language: `web/chN_ro.html` (RO) and `web/chN.html` (EN).
The page is generated from the lecture deck by `latex/build_web_lecture.py` (ported from the MFM course). Without course text the
page shows the slide content as HTML (the skeleton). With course text, each frame gets textbook prose, and the slide content
moves into a compact „Pe scurt” / "In brief" box under it.

Finished example of the target (MFM course, read only, do not edit):
`/Users/danpele/Documents/2026 MFM/MFM/latex/web_text/ch4_ro.md` and `ch4_en.md`, rendered in `.../MFM/web/ch4_ro.html`.

## Files
| What | Where |
|---|---|
| Course text you write | `latex/web_text/chN_ro.md`, `latex/web_text/chN_en.md` (create the folder if missing) |
| Decks (read only) | the ones the site links in `assets/course-data.js`: `RO/Cursuri/capitolN_*.tex`, `EN/Courses/chapterN_*.tex`. **EN/Courses also holds older decks with the same chapter number** — `python3 -c "import build_web_lecture as b; print(b.site_decks()[N])"` (in `latex/`) prints the right pair |
| Generated pages | `web/chN_ro.html`, `web/chN.html`, `web/slides/` (fallback slide images), `web/img/` (PDF charts as PNG), log `web/chN_log.txt` |
| Layout / behaviour | `web/web.css`, `web/web.js` (shared by all chapters; do not edit without telling the other agents) |

Never edit the decks, `build_chapterN.py`, `chN_common.py`, `build_web_lecture.py`, `web.css`, `web.js`, `index*.html` or
`assets/*`. If the deck is wrong (a number, a typo), report it; do not work around it in the text. Do not add buttons or links
to the course site. No git commit / push.

## Build and check
```
cd 2026/latex
python3 build_web_lecture.py N            # writes web/chN_ro.html, web/chN.html and web/chN_log.txt; prints the stats and the log
python3 build_web_lecture.py N --hashes   # every frame: "ro p. 12: ## <key> {h=<hash>}" — copy these lines as your ## headings
python3 build_web_lecture.py N --keys     # citation keys of the deck: Key  \refMacro  DOI
python3 build_web_lecture.py all          # every chapter
python3 refresh_web_text_keys.py N        # after a deck change: updates the {h=…} of your headings (and renamed titles)
```
Check in headless Chrome (server from the repo root: `cd 2026 && python3 -m http.server 8765`):
```
cd 2026/web && ../latex/check_interactive_page.sh chN_ro.html web        # and chN.html; add a width as 3rd argument
```
It must print `[]` (no JS errors), `mj=done mjerr=0` (MathJax rendered, no TeX errors) and `w=1440/1440` (no horizontal
overflow). For 390 px, load the page in an iframe 390 px wide (headless Chrome does not go below 500 px) and check that no
element ends right of 390 px. Then look at the page: full-page screenshot (`--window-size=1440,12000 --screenshot=…`), crop it
and read every part you changed. The log `web/chN_log.txt` must have no `text out of date`, `text without a matching frame`
or `text key to rename` lines when you finish.

## Format of `latex/web_text/chN_{ro,en}.md`
```
<!-- comment: ignored -->

## @intro                                (optional; before the first #) chapter opener, see below

# <section title, verbatim>               text up to the first ## = introduction of the section
Paragraphs …

## <frame title, verbatim> {h=<hash>}      prose of that frame (2–3 paragraphs)
Paragraphs …
@after                                   (optional line) paragraphs below it go after the frame's live block (interactive pages)

## @end                                  synthesis of the current section, after its last frame
Paragraphs …
```
- Section titles and frame keys must match the deck **exactly**: copy them from `--hashes` (frame keys) and from the
  `\section{…}` lines (plain text: `S\&P 500` is `S&P 500`, `VaR 1\%` is `VaR 1%`). A frame title that occurs twice gets the
  key `<title> #2`; a frame without a title has the key `p<page>`. An unmatched heading is reported in the log
  (`text without a matching frame`) and its text is not shown.
- The hash `{h=…}` records which version of the frame you described. When the deck changes, the build logs
  `text out of date: frame p. N`; reread the frame, update the prose, then run `refresh_web_text_keys.py N`.
- Which frames get prose: every frame of every main section (from the first `\section` to the summary section). Front matter
  before the first section (title page, acronyms, „Întrebarea de azi…”, „Bibliografie și instrumente”) needs none: the opener
  `## @intro` replaces it. The deck's bibliography section is hidden on the web page and replaced by a generated reference list.
- Paragraphs are separated by blank lines. Inline: `$…$` math (MathJax; the deck macros `\E`, `\Var`, `\Cov`, `\R` … work),
  `**bold**`, `*italic*`, `[text](url)`.
- Display math: a paragraph `$$ … $$`. Numbered: `$$ … $$ {#eq:label}` → numbered (N.k) in page order; refer to it with
  `@eq:label`. Labels and numbering must be identical in RO and EN (the build stops otherwise).
- Citations: `@Key` → „Engle (1982)”, `[@Key]` → „(Engle, 1982)”, `[@Key1; @Key2]`, or `[[refEngle]]` (the deck macro). Keys come
  from the deck's `\refX` macros that have an entry in its bibliography (`--keys` lists them; e.g. `\refFHH` → `FrankeHardleHafner2019`,
  `Pele et al.\ (2023)` → `PeleEtAl2023`). An unknown key stops the build. RO and EN must cite the same set of works.
- Links inside the page: sections `#sec-K` (K = 1, 2, … in page order), frames `#s<page>` (page = PDF page of the frame, the
  `p.` number of `--hashes`), deck targets `#h-<name>`. Other chapters: `chM_ro.html` / `chM.html` (same folder).
- Charts: the slide's chart is already in the „Pe scurt” box; do not add images in the text.

### Chapter opener `## @intro`
Directives, one per line, all optional, in any order; the text belongs to the directive above it:
```
## @intro
@epigraph <Key> [| <title>]       a short verbatim quotation (≤ ~35 words) from a work of the deck's bibliography, as "> …" lines,
                                  in the original language (English) on both pages; verified word for word against an accessible
                                  copy of the original; never from memory. No verifiable quotation, no epigraph.
@portrait <file> | <name> | <credit> | <URL>   portrait of the quoted author (photos/…; licence verified on its Wikimedia Commons page, URL = that page)
@text                             2–3 paragraphs: why the chapter's question matters, only from the chapter's own slides and results
@chart <image base name>          opening chart (an image of charts/, by base name, or the anchor of a live block); lines below = caption
@goals                            "- " list of what the reader will be able to do (from the deck's learning outcomes)
@plan                             "- <section title, verbatim> | <one-line description>" per section, in order
```

## Content rules
- **Textbook prose for every frame**: 2–3 paragraphs that teach the frame's content — definitions, the reasoning behind a formula,
  what a result means and why. The slide's bullets stay in the „Pe scurt” / "In brief" box; the prose must not be a bullet list in
  sentences, and it must be readable without the box.
- Frame and section titles are the deck's, verbatim (they come from the deck; your headings only select the frame).
- No new claims or numbers: every number comes from the slides or the chapter's results (Quantlets/Ch_NN results JSON). Every
  number says what it is („deviația standard zilnică a S&P 500 în 2017: 0,42%”, not „0,42”). No lists of bare numbers.
- No AI tone: no slogans, no promotional openings, no punchlines, no rhetorical questions as section ends, no „Let's dive in”,
  no „crucial/key insight/powerful”.
- No process remarks: never „verified in Quantlet”, „exactly as in the paper”, „same numbers as the package”, „we checked”,
  „we dropped”, „the code computes…”. No computation-report tone: teach the concept and the finding, with the data as evidence.
- Never write „not examined” / „nu se examinează”: say what the chapter does, not what it leaves out.
- No session or duration notes („Ședința 1”, „80 de minute”); no exam mentions.
- Research-grade level (the audience is a finance/statistics class that has had probability and econometrics): no beginner
  explanations of what a mean is, but define every acronym and metric at first use on the page.
- Seminars: never reveal or paraphrase seminar solutions; the seminar sources are private. You may link the student seminar PDF
  (`../RO/Seminarii/seminarN_…_ro.pdf`, `../EN/Seminars/seminarN_….pdf`, as in `assets/course-data.js`), nothing else.

### Language
- RO orthography: „sînt” (not „sunt”), „î” inside words („cîștig”, „hotărî”), „â” only in the „român” family; „nicio/niciun”.
- „distribuția Normală” (never „Normala” or „the Normal” as a noun); in EN „the Normal distribution”.
- „media” (not „speranța”; „speranța matematică” only where the deck's own title uses it, since titles are verbatim).
- English financial jargon stays in English in RO text, without calques: drawdown, volatility clustering, backtesting, leverage
  effect, out-of-sample, shrinkage, bootstrap, spread, benchmark …
- At the first use of a Romanian technical term, give the English term in italics in parentheses:
  „grupare a volatilității (*volatility clustering*)”, „cozi groase (*heavy tails*)”, „funcția de repartiție (*distribution function*)”.
- VaR convention: say „VaR 1%” / „ES 2,5%” (never „VaR 99%”); VaR_α = −q_α, the negative of the α-quantile of returns.
- RO number format: decimal comma (0,42%), thin/no-break space for thousands (6 714); EN: decimal point (0.42%), comma for thousands.
- Questions to the reader in RO: „Întrebare pentru sală”, „Ce credeți?”; never „preziceți”.
- RO and EN are the same text in two languages: same paragraphs, numbers, citations and equations.

## Skeleton status (phase 1, 1 Nov 2026)
All 34 pages (chapters 0–16, RO and EN) build with 0 JS errors, MathJax rendered, no horizontal overflow at 1440 and 390 px.
Frames shown as slide images (TikZ diagrams, the converter cannot render them): ch0 p. 16, 18, 37 (RO and EN);
ch16 RO p. 8 / EN p. 7 (course map). Prose for these frames goes above the image as usual.
