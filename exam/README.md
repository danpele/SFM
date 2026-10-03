# SFM exam materials

Exam materials of *Statistica piețelor financiare / Statistics of Financial Markets* (bachelor, year 3, CSIE, ASE).
All documents compile with XeLaTeX (system font Helvetica Neue) and use the shared preamble `sfm_exam_preamble.tex`
(ASE and IDA logos, automatic numbering of problems and sub-tasks, the `\ifrez` switch for solutions).

| Path | Content | In git |
|---|---|---|
| `sfm_exam_preamble.tex` | Shared article preamble (RO/EN through `\def\examlang{ro|en}`) | yes |
| `make_figs.py` | Every exam chart, in the course style, from `data/market` via `Quantlets/common/sfm_data.py` and `sfm_style.py` | yes |
| `build_variant.py` | Assembles an exam variant from the bank and compiles it | yes |
| `2026/` | The June 2026 exam (copy of `Probleme examen/2026_LaTeX_source`, which stays in place) | wrappers, header, charts and problem PDFs yes; `body_v*.tex` (solutions inline) and `*_Rezolvare.*` no |
| `practice/` | Public practice set without solutions (answers only): `probleme_examen_ro.pdf`, `exam_problems_en.pdf` | yes |
| `bank/` | Instructor-only problem bank: `ro/chNN.tex`, `en/chNN.tex`, `figs_chNN.py`, `figs/`, `bank_*.pdf` | **no** |
| `variants/` | Assembled variants, problems and solutions | **no** |

Solutions are never committed. `exam/.gitignore` ignores `bank/`, `variants/`, `2026/body_v*.tex` and every `*_rezolvare.*`,
`*_Rezolvare.*`, `*_solutions.*` and `*_barem.*` file under `exam/`. Keep the bank and the solution PDFs locally
(or in the instructor folder outside the repository).

## Building an exam variant

```bash
python3 exam/make_figs.py                                        # charts (2026 exam, practice set, bank)
python3 exam/build_variant.py --list                             # problems in the bank
python3 exam/build_variant.py --name V1 --seed 2027 --n 9        # 9 problems from 9 different chapters, RO
python3 exam/build_variant.py --name V1 --seed 2027 --n 9 --lang both --date "iunie 2027"
python3 exam/build_variant.py --name R1 --ids ch01-p2,ch05-p1,ch10-p4 --lang both   # chosen problems
python3 exam/build_variant.py --name V2 --seed 11 --chapters 1-10                   # chapters 1..10 only
python3 exam/build_variant.py --bank --lang both                 # the whole bank with solutions
```

- The same seed always gives the same variant; RO and EN use the same problem ids.
- Each bank problem is worth 1 point (two sub-tasks of 0.5 points, step-by-step solution and grading scheme).
  With `--n 9` a variant has 9 points + 1 point by default = 10 points, as in June 2026.
- Output: `exam/variants/<name>_ro.pdf` and `<name>_ro_rezolvare.pdf`, `<name>_en.pdf` and
  `<name>_en_solutions.pdf`. Each document is compiled twice; the script reports errors and overfull boxes.

## Problem format in the bank

```latex
%<problem id=ch05-p2 pts=1>
\problem[ch05-p2]{Title}{1p}
Context with the data.
\Q First sub-task (one question). \pts{0,5p}
\Q Second sub-task (one question). \pts{0,5p}
\ifrez\Rez
\begin{enumerate} \item step 1 \item step 2 \end{enumerate}
\barem{0,25p ...; 0,25p ...; 0,5p ...}
\fi
%</problem>
```

A chapter figure is produced by `exam/bank/figs_chNN.py` (function `make(out_dir)`), which `make_figs.py` runs and
whose numbers it saves in `bank/bank_numbers.json`.

## Practice set

`practice/probleme_examen_ro.tex` and `practice/exam_problems_en.tex`: 26 problems from chapters 1--14, different
from the bank problems, with numerical answers at the end. Recompile after `python3 exam/make_figs.py --only practice`.
Site links: `https://danpele.github.io/SFM/exam/practice/probleme_examen_ro.pdf` and
`https://danpele.github.io/SFM/exam/practice/exam_problems_en.pdf`.
