# SFM build pipeline

How the materials of *Statistica piețelor financiare / Statistics of Financial Markets* are built. The pipeline is adapted from the MFM course: same slide design, same bilingual generators, same data and notebook rules. Run every command from the repository root.

## Layout

| Path | Content |
|---|---|
| `latex/preamble.tex` | Shared Beamer preamble. It defines the logos, `\quantlet`, `\sfmquantlet`, `\colaburl`, `\nb`, `\itemsize`, `\imgcredit`, `\imgcap`, `\ifsolutions`/`\solonly`/`\propsub`, and the appendix and chapter-link macros. |
| `latex/sfm_chapters.py` | Chapter registry (titles from `assets/course-data.js`) and the file-naming scheme. |
| `latex/sfm_build.py` | Common generator framework: `⟦EN‖RO⟧` text, `@{key}` numbers, the `Deck` class, legacy conversion, compilation. |
| `latex/build_chapterN.py`, `latex/build_seminarN.py` | One generator per deck. |
| `latex/acronyms.py`, `_acr_scan.py`, `acronyms_extra/chN.py` | Acronym glossary, inserted after the title page. |
| `latex/appendix_links.py` | Appendix buttons and back-buttons; "Chapter N" mentions become links to the PDF on the site. |
| `data/market/*.csv`, `data/manifest.csv` | Daily market data from EODHD, saved once (91 series, copied from MFM, ending 18.09.2026). |
| `Quantlets/common/sfm_data.py` | Data loader: reads the local file, or the raw GitHub URL of the SFM repository. EUR/RON comes from the BNR reference rate (online). No API keys. |
| `Quantlets/common/sfm_style.py` | Chart style: transparent background, legend below the plot, MFM palette, no grey; `check_no_grey`. |
| `Quantlets/common/sfm_quantlets.py` | Quantlet builder: `Metainfo.txt` plus a self-contained Colab notebook plus charts. |
| `Quantlets/Ch_NN/` | Per chapter: `generate_all_charts.py` (charts and tables), `build_quantlets.py`, `SFM_chN_*` folders. |
| `notebooks/sfm_notebook.py`, `notebooks/build_notebooks_chN.py` | Notebook builders. Output is English only, in `notebooks/EN/`. |
| `notebooks/add_colab_banner.py` | Adds the "Save a copy in Drive" banner and the optional Drive-save cell. |
| `notebooks/split_seminar_notebooks.py`, `split_quantlet_seminars.py` | Split each seminar notebook and Quantlet into a public student version and a private instructor version. |

### File names

Slugs are derived from the chapter titles in `assets/course-data.js`. To list them all, run `python3 latex/sfm_chapters.py`.

| Material | Path |
|---|---|
| Lecture, EN | `EN/Courses/chapterN_<slug_en>.pdf` |
| Lecture, RO | `RO/Cursuri/capitolN_<slug_ro>.pdf` |
| Seminar, EN | `EN/Seminars/seminarN_<slug_en>.pdf` (instructor version: `…_solutions.pdf`, git-ignored) |
| Seminar, RO | `RO/Seminarii/seminarN_<slug_ro>_ro.pdf` (instructor version: `…_solutions.pdf`, git-ignored) |
| Notebooks | `notebooks/EN/chapterN_lecture_notebook.ipynb`, `notebooks/EN/chapterN_seminar_notebook.ipynb` |
| Site | `https://danpele.github.io/SFM/<path>` |

Glossary and chapter-link processing only touches decks that sit at these paths **and** contain `\input{../../latex/preamble}`. The old decks in `RO/Courses`, `RO/Seminars` and `EN/Courses` (old names) stay as they are until their chapter is rebuilt.

## Building a chapter (commands in order)

```bash
# 1. charts, tables and numbers (Quantlets/Ch_NN)
python3 Quantlets/Ch_NN/generate_all_charts.py
python3 Quantlets/Ch_NN/seminarN.py                    # if the chapter has seminar computations
# 2. Quantlet folders (Metainfo.txt + Colab notebook + charts)
python3 Quantlets/Ch_NN/build_quantlets.py
# 3. decks EN + RO (each generator also runs latex/acronyms.py N, which runs appendix_links.py)
python3 latex/build_chapterN.py
python3 latex/build_seminarN.py                        # also writes the *_solutions.tex wrappers
# 4. compile: pdflatex twice per deck (and per _solutions wrapper); prints errors and overfull vbox
python3 latex/sfm_build.py compile N
# 5. notebooks (EN only), then execute them
python3 notebooks/build_notebooks_chN.py
jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapterN_lecture_notebook.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapterN_seminar_notebook.ipynb
# 6. Colab banner + Drive-save cell (new-pipeline notebooks and Quantlets only; idempotent)
python3 notebooks/add_colab_banner.py
# 7. seminars: student version public, full version private (../instructor/Quantlets/Ch_NN)
python3 notebooks/split_seminar_notebooks.py N         # also runs split_quantlet_seminars.py N
python3 notebooks/split_quantlet_seminars.py --write-gitignore   # private answer charts -> .gitignore block
python3 notebooks/split_quantlet_seminars.py --check   # nothing private in public folders or in git
```

Re-run step 7 after **any** `build_quantlets.py`, because that script rewrites the full seminar notebooks.

Before publishing, update the chapter's links in `assets/course-data.js`: slides, seminar, the Colab notebook and the Quantlet folder.

### Converting an existing deck instead of rewriting it

```bash
python3 latex/sfm_build.py convert <old.tex> N en|ro [lecture|seminar]
```

The converter keeps everything after `\begin{document}`. It replaces the inline preamble and title with the shared preamble and the standard title, and writes the deck under its new name. Chapter 0 is built this way by `latex/build_chapter0.py`, which also applies small text replacements.

## Writing a generator

`latex/sfm_build.py` has a worked example in its docstring. The rules:

- Write all text as `⟦english||română⟧`.
- Never type numbers by hand. Use `@{key}` with `Values.put(key, x, decimals)`.
- In RO decks, decimal points become commas and „de” is added after numerals of 20 and above (for example, „131 de zile”).
- Lecture helpers: `D.section`, `D.frame`, `D.chart` (chart plus Quantlet link), `D.recap`, `D.references`.
- Seminar helpers: `D.solved` (task and solution, visible to everyone), `D.proposed` (solution only with `\solutionstrue`), `D.task` (an explicit task with numbered sub-tasks and what to report), `D.frame(..., instructor_only=True)`.
- Images: `photo(file, caption, url, credit)` adds a visible credit. Citations are `\href` links to the DOI.
- Chapter-specific acronyms go in `latex/acronyms_extra/chN.py`. The script prints `NOT IN DICTIONARY` for any acronym that is missing.

## Data

- The source is daily market data from EODHD, saved once in `data/market`. Do not call any API.
- Convention for which price column to use:
  - Indices, FX, crypto and yields use `close`.
  - ETFs and stocks use `adjusted_close`.
- Weekend quotes and unchanged (holiday-filled) closes are dropped, except for crypto.
- Annualise with the actual observation frequency of each series.
- EUR/RON is the official BNR reference rate (`sfm_data.load_close('eurron')`). The EODHD EUR/RON series contains erroneous quotes.
- New code reads data only through `sfm_data.py`. The old Quantlets still use yfinance; they will be replaced chapter by chapter.

## Chapter 0 (pipeline check)

```bash
python3 Quantlets/Ch_00/generate_all_charts.py && python3 Quantlets/Ch_00/build_quantlets.py
python3 latex/build_chapter0.py && python3 latex/sfm_build.py compile 0
python3 notebooks/build_notebooks_ch0.py && python3 notebooks/add_colab_banner.py
```
