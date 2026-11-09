r"""
sfm_opening.py -- deschiderea seminariilor SFM 9--12 după modelul seminarului 2 MFM (fără redundanțe)
=====================================================================================================
Un singur cadru „Întrebarea de azi și structura seminarului” (întrebarea, rezultatele de obținut, notebook-ul,
contactul) și harta exercițiilor; tipurile A/B/C și marcajele [Rezolvat]/[Propus] se explică o singură dată,
pe hartă. Butoanele „Lista exercițiilor” (latex/exercise_links.py) duc la hartă.

Utilizare într-un generator:
    from sfm_opening import opening_frame, MAP_LEGEND
    opening_frame(D, T('question?', 'întrebare?'))
    D.frame(T('Exercise Map', 'Harta exercițiilor'), table(...) + items(*MAP_LEGEND))
"""

from sfm_build import items
from sfm_route import CONTACT, NOT_GRADED

MAP_LEGEND = (
    '⟦\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it '
    'after the model; the solution is discussed in the seminar'
    '||\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați '
    'după model; soluția se discută la seminar⟧',
    '⟦A: calculations on paper; B: real data, in the notebook, each ending with an interpretation question; C: open questions, '
    'among them the audit of an AI answer'
    '||A: calcule pe hîrtie; B: date reale, în notebook, fiecare încheiat cu o întrebare de interpretare; C: întrebări deschise, '
    'printre care verificarea unui răspuns AI⟧')


def opening_frame(D, question, has_c1=True):
    """The single structure frame: question, outputs, notebook, contact (no exercise list: that is the map)."""
    c_out = ('⟦a short argued answer to C1; the errors found in the AI answer of C2 and the corrected answer'
             '||un răspuns scurt și argumentat la C1; erorile găsite în răspunsul AI din C2 și răspunsul corectat⟧' if has_c1 else
             '⟦the errors found in the AI answer of C2 and the corrected answer||erorile găsite în răspunsul AI din C2 și răspunsul corectat⟧')
    D.frame('⟦Today\'s Question and Seminar Structure||Întrebarea de azi și structura seminarului⟧', items(
        ('⟦Question of the seminar||Întrebarea seminarului⟧', [f'\\textbf{{{question}}}']),
        ('⟦Outputs to produce||Rezultate de obținut⟧',
         ['⟦the answers to the A exercises||răspunsurile la exercițiile A⟧',
          '⟦the table or figure each B exercise asks for||tabelul sau figura cerute la fiecare exercițiu B⟧',
          c_out,
          '⟦the interpretation sentences each exercise asks for||frazele de interpretare cerute la fiecare exercițiu⟧']),
        NOT_GRADED,
        ('⟦Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}'
         '||Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}⟧',
         ['⟦each task names its notebook section||fiecare cerință indică secțiunea din notebook⟧']),
        '⟦Exercises, their types and their order: the Exercise Map (next slide)'
        '||Exercițiile, tipurile și ordinea lor: harta exercițiilor (slide-ul următor)⟧',
        CONTACT))
