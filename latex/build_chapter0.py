r"""
build_chapter0.py -- Capitolul 0 (Introducere), EN + RO, in noul flux SFM
=========================================================================
Etapa de infrastructura: continutul este cel existent (deck-urile din primavara 2026), convertit la preambulul
comun (latex/preamble.tex), cu titlul standard, numele noi si glosarul de acronime. Cind capitolul va fi rescris,
acest fisier devine un generator bilingv complet (Deck din latex/sfm_build.py, text ⟦EN||RO⟧).
Surse:
  EN  EN/Courses/chapter0_introduction.tex          (vechiul deck EN; conversia este idempotenta)
  RO  RO/Courses/20260310_chapter0_introduction_ro.tex
Iesire:
  EN/Courses/chapter0_introduction.tex
  RO/Cursuri/capitol0_introducere.tex
Rulare:  python3 latex/build_chapter0.py   apoi   python3 latex/sfm_build.py compile 0
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_build import ROOT, convert_legacy, run_acronyms   # noqa: E402

SRC = {'en': 'EN/Courses/chapter0_introduction.tex',
       'ro': 'RO/Courses/20260310_chapter0_introduction_ro.tex'}
REPLACE = {
    'en': [('\\vspace{0.8cm}\n\n        \\begin{block}{Seminar Grade}', '\\vspace{0.4cm}\n\n        \\begin{block}{Seminar Grade}')],   # overfull vbox
    'ro': [('Statistica pe Piețe Financiare', 'Statistica piețelor financiare')],
}

if __name__ == '__main__':
    for lang, rel in SRC.items():
        convert_legacy(os.path.join(ROOT, rel), 0, lang, replace=REPLACE[lang])
    run_acronyms(0)
