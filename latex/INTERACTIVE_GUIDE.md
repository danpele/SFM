# Interactive chapter pages (SFM) — how to build chapter N

The infrastructure is ported from the MFM course. SFM has no finished chapter yet, so the reference implementation is
**MFM Chapter 4** (read it, never edit it; paths below are absolute):
- `/Users/danpele/Documents/2026 MFM/MFM/interactive/ch4_ro.html` (the page) and `.../MFM/interactive/ch4.js` (one render function per block, `init()` wires the inputs and the ▶ buttons).
- `/Users/danpele/Documents/2026 MFM/MFM/latex/build_interactive_ch4.py` — the chapter spec (BLOCKS, PFX, SECTIONS, L, widgets, QA, TEXT_JS, PREDICT, TICKET).
- `/Users/danpele/Documents/2026 MFM/MFM/Quantlets/Ch_04/export_interactive.py` — writes `interactive/ch4_data.js` by calling the chapter's own chart code.
- MFM uses `window.MFMI` / `window.MFM_LANG`; in SFM the same objects are **`window.SFMI`** / **`window.SFM_LANG`** (and the MathJax event of the web pages is `sfm-mathjax`). Rename them when you copy code.

SFM files (all paths relative to the repo root `2026/`):
- `interactive/common.js` — shared helpers (`window.SFMI`): linear algebra, seeded Normal draws, Plotly layout, `Player` (▶ button), `predict`, `exitTicket`, formatting. **Do not edit it**; put chapter helpers in `interactive/chN.js`. If a helper belongs in common.js, say so in your report.
- `interactive/interactive.css` — page styles (do not edit).
- `latex/interactive_common.py` — builds the pages and slide images from a spec. Do not edit it.
- `latex/check_interactive_page.sh` — headless check (below).
- `latex/build_web_lecture.py` — the web lecture; it picks up `build_interactive_chN.py` automatically and replaces each chart of BLOCKS by its live block (see `latex/WEB_GUIDE.md`).

## What a chapter page is
- One block per **data chart in the lecture body** (before `\section{AI pentru descoperire științifică}` / AI for Scientific Discovery), in slide order, grouped by the lecture's sections.
  Skip portraits, photos, historic images and figures copied from papers. If a chart cannot be made interactive in a meaningful way, still give it a block where the user can at least hover and change one meaningful parameter (period, asset, threshold); never skip silently — list skipped charts in the report with the reason.
- Left: the slide image (built automatically from the .nav file). Right: the same figure computed in the browser. **Its initial state reproduces the slide chart** (same data, same parameters, same numbers).
- Every block has a ▶ button (`Player`) that moves one meaningful parameter step by step (a window, a threshold, time, the number of simulations, a confidence level, a GARCH parameter …). Choose the parameter that shows the mechanism of the slide.
- Under each block: one "Întrebare pentru sală" / "Question for the room" and its "Răspuns" / "Answer", taken from what the slide teaches, with the numbers of the chapter.
- RO page `interactive/chN_ro.html` and EN page `interactive/chN.html` (texts in the spec; chart text always in English, in both pages).

## Steps
1. Read the chapter's RO and EN lecture .tex (body only) and `Quantlets/Ch_NN/` (chart code + results JSON). The decks are the ones the site links (`assets/course-data.js`): `RO/Cursuri/capitolN_*.tex`, `EN/Courses/chapterN_*.tex`. **EN/Courses also holds older decks with the same chapter number** — never pick a deck by file name.
2. Write `Quantlets/Ch_NN/export_interactive.py`: call the chapter's own functions; export only what the page needs (moments, derived series, model outputs, test results). Re-use results JSON numbers; do not reimplement the chapter's estimation differently. Reading `data/` files of the repo is fine; never call a data API. Do **not** run any `build_quantlets.py` (it regenerates every Quantlet folder).
3. Write `interactive/chN.js` (import from `window.SFMI`) and `latex/build_interactive_chN.py` (copy the structure of the MFM ch4 spec). In the spec:
   ```python
   from build_web_lecture import site_decks
   N = 9
   TEX = site_decks()[N]                      # {'ro': 'RO/Cursuri/capitol9_...', 'en': 'EN/Courses/chapter9_...'}
   QL = 'https://github.com/danpele/SFM/tree/main/Quantlets/Ch_09'
   ```
   `L[lang]` needs `course` ('Statistica piețelor financiare' / 'Statistics of Financial Markets'), `site` ('../index_ro.html#chapters' / '../index.html#chapters') and the other keys of the MFM spec. Keep `build(sys.modules[__name__])` under `if __name__ == '__main__':` so that build_web_lecture.py can import the spec without side effects.
4. `cd 2026/latex && python3 build_interactive_chN.py` (writes pages, readouts, slide images), then `python3 build_web_lecture.py N` (the web lecture gets the live blocks).
5. Check (server: `cd 2026 && python3 -m http.server 8765`):
   `cd 2026/interactive && ../latex/check_interactive_page.sh chN_ro.html` and the same for `chN.html`;
   `cd 2026/web && ../latex/check_interactive_page.sh chN_ro.html web` (web lecture). Must print `[]` errors, `empty=` nothing, plots > 0, `mj=done mjerr=0` on web pages, and page width = window width.
6. Verify numbers: compare the page's initial readouts with the results JSON / slide numbers (small node script: load `chN_data.js` + `common.js` + `chN.js` with `window`/`document` stubs and call the exported functions).
7. Look at it: headless Chrome full-page screenshot (`--window-size=1440,12000 --screenshot=...`), crop and inspect every block: labels must not overlap, legends below the chart, nothing cut. Also check 390 px (an iframe of width 390 in a wrapper page: headless Chrome does not go below 500 px).

## Rules
- Language (both pages): no AI tone. No slogan titles, no promotional intro, no punchlines, no meta remarks about the page ("every number is recomputed…"). Block titles = slide titles (automatic). Readouts state results plainly, with numbers that say what they are.
- RO orthography: „sînt”, „î” inside words, „â” only in the „român” family; „distribuția Normală”; „media” (not „speranța”); English finance jargon stays English (drawdown, volatility clustering, backtesting …). Never „preziceți”. VaR convention: „VaR 1%”, VaR_α = −q_α (never „VaR 99%”).
- Charts: transparent background, legend outside at the bottom, no grey series or grey text (use navy #1F2A44 for reference lines), the chapter's own series colours (take them from its chart code in Quantlets/Ch_NN), English text.
- Data: never substitute another series and never rescale or adjust values to match the lecture: numbers must match because the computation is the chapter's own. If something cannot be reproduced, say so in the report.
- Data and numbers must match the lecture. If the page shows something the slide does not (e.g. a simulation), its default must still agree with the chapter's numbers.
- Do not modify lecture decks, lecture .tex, lecture builders (`build_chapterN.py`, `chN_common.py`), seminar files, `common.js`, `interactive_common.py`, `build_web_lecture.py`, `site.css`, `course-data.js`, or other chapters' files. Seminar sources (`EN/Seminars/*.tex`, `RO/Seminarii/*.tex`, `build_seminar*.py`) are private: never copy solutions into the pages and never link them.
- Do not add buttons or links to the course site (index.html, course-data.js). No git commit / push.

## Writing predictions and exit-ticket questions
Opt-in helpers in `common.js` (`SFMI.predict`, `SFMI.exitTicket`); reference spec: `PREDICT`, `TICKET`, `QA` in the MFM `build_interactive_ch4.py`. Wire with `predict(block, { ..., player, veil: { out: true } })`: the slide and the readout stay blurred until „Arată slide-ul”, „Verifică” or the end of ▶.
- The question must not be answerable by reading the page. If the slide or the initial readout prints the answer, ask about what ▶ will do (direction or size of the change as the parameter moves), or keep the question and use `veil`.
- If the initial live chart (or its hover) shows the answer, add `charts: true` (all `.chart` divs of the block) or `charts: ['id', …]` to the veil. Leave it off when the question asks what ▶ will do and the initial chart does not show it.
- Ask exactly one thing, and say which object is meant ("in this simulation", "the first tested window", "the GARCH(1,1) fitted to the S&P 500").
- Every wrong option encodes one identifiable misconception (forgetting the square, adding volatilities instead of variances, √T scaling where it does not hold, p-value as the probability that the effect is real, no evidence = equivalence, VaR 1% read as the 99% quantile of returns). No arbitrary numbers and no options the slide contradicts outright.
- Prefer numeric distractors that a real mistake produces and say in the feedback which mistake gives them.
- One idea per option; all options answer the question asked and have similar length and form.
- The key is verified against what the page computes (node script on `chN_data.js` + `common.js` + `chN.js`), not against memory.
- If the page computes something different from the slide (fewer simulations, another bootstrap), the answer gives both numbers and says which is which.
- Feedback explains why, with numbers that say what they are.
- Inference: use the test the question asks for (one-sided question → one-sided critical value); "not significant" never means "equal" or "no structure", only "not separated from noise by this test".
- Asymptotic bands and thresholds are reference values, not exact finite-sample guarantees; word conclusions accordingly.
- Exit ticket: 3–5 questions taken from the chapter's quick-check / summary slides; every option gets a "why", the correct one included.
- RO: „Ce credeți?”, „Alegeți”, never „preziceți”; no exam mentions, no AI tone.

## Report (final message)
Files created, number of blocks, skipped charts with reason, checks output (RO and EN, interactive and web pages), numbers verified against which results, anything left open.

## Web course text
See `latex/WEB_GUIDE.md` (format of `latex/web_text/chN_{ro,en}.md`, content rules, build and checks).
