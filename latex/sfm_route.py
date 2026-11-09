r"""
sfm_route.py -- cadrele „Întrebarea de azi și structura seminarului (i/k)” ale seminariilor SFM, EN + RO
=========================================================================================================
Model: seminarul 1 MFM. Fiecare cod de exercițiu apare împreună cu un titlu scurt, luat din cadrul exercițiului.
  (1/k)      întrebarea (întrebările) seminarului, cele trei tipuri de exerciții, marcajele [Rezolvat]/[Propus]
  (2..k-1)   exercițiile de azi, în ordine, grupate pe părți; o listă lungă se împarte pe mai multe cadre
  (k/k)      rezultatele de obținut, notebook-ul, contactul
Nu se scrie nicăieri că seminarul are loc înaintea cursului (decizia titularului).

Utilizare într-un generator:
    from sfm_route import route_frames, PART_A, PART_B, PART_C, C2_AI
    route_frames(D, question, [[(PART_A, [('A1', T('...', '...')), ...])],
                               [(PART_B, [...]), (PART_C, [...])]])
"""

from sfm_build import items

PART_A = '⟦Part A: calculations on paper||Partea A: calcule pe hîrtie⟧'
PART_B = '⟦Part B: real data, in the notebook||Partea B: date reale, în notebook⟧'
PART_C = '⟦Part C: open questions||Partea C: întrebări deschise⟧'
C2_AI = ('C2', '⟦audit an AI answer and correct its errors||verificarea unui răspuns generat de AI și corectarea erorilor⟧')

# Titularul de seminar al SFM nu este încă stabilit (assets/course-data.js: TODO_SEMINAR_INSTRUCTOR).
CONTACT = ('⟦Contact: Prof.\\ Daniel Traian Pele, \\href{mailto:danpele@ase.ro}{danpele@ase.ro}'
           '||Contact: prof.\\ Daniel Traian Pele, \\href{mailto:danpele@ase.ro}{danpele@ase.ro}⟧')

NOT_GRADED = ('⟦Nothing is handed in: the seminar is for practice; the solutions of the [Proposed] exercises are discussed in class'
              '||Seminarul are rol de exercițiu și nu se notează; rezolvările exercițiilor [Propus] se discută la seminar⟧')


def _title(k, n, plural):
    if plural:
        return f'⟦Today\'s Questions and Seminar Structure ({k}/{n})||Întrebările de azi și structura seminarului ({k}/{n})⟧'
    return f'⟦Today\'s Question and Seminar Structure ({k}/{n})||Întrebarea de azi și structura seminarului ({k}/{n})⟧'


def route_frames(D, question, pages, questions=None, has_c1=True, not_graded=NOT_GRADED):
    """question: ⟦EN||RO⟧ text (bold); questions: a list of questions instead of one.
    pages: one list per frame of (part heading, [(code, title), ...])."""
    n = len(pages) + 2
    plural = questions is not None
    first = (('⟦Questions of the seminar||Întrebările seminarului⟧', [f'\\textbf{{{q}}}' for q in questions]) if plural
             else ('⟦Question of the seminar||Întrebarea seminarului⟧', [f'\\textbf{{{question}}}']))
    D.frame(_title(1, n, plural), items(
        first,
        ('⟦Three types of exercises||Trei tipuri de exerciții⟧',
         ['⟦\\textbf{A}: calculation exercises, on paper||\\textbf{A}: exerciții de calcul, pe hîrtie⟧',
          '⟦\\textbf{B}: exercises on real data, solved in the notebook||\\textbf{B}: exerciții pe date reale, rezolvate în notebook⟧',
          '⟦\\textbf{C}: open questions, e.g.\\ correcting an answer given by an AI tool'
          '||\\textbf{C}: întrebări deschise, de exemplu corectarea unui răspuns dat de un instrument AI⟧']),
        ('⟦Each exercise is marked||Fiecare exercițiu este marcat⟧',
         ['⟦\\textbf{[Solved]}: the full solution is presented in the seminar||\\textbf{[Rezolvat]}: rezolvarea completă este prezentată la seminar⟧',
          '⟦\\textbf{[Proposed]}: you solve it; the solution is discussed in the seminar||\\textbf{[Propus]}: îl rezolvați dumneavoastră; soluția se discută la seminar⟧'])))
    for k, page in enumerate(pages, 2):
        groups = ['⟦Today\'s exercises, in the order in which they are solved||Exercițiile de azi, în ordinea în care se rezolvă⟧'] if k == 2 else []
        groups += [(head, [f'{code}: {title}' for code, title in tasks]) for head, tasks in page]
        D.frame(_title(k, n, plural), items(*groups))
    c_out = ('⟦a short argued answer to C1; the errors found in the AI answer of C2 and the corrected answer'
             '||un răspuns scurt și argumentat la C1; erorile găsite în răspunsul AI din C2 și răspunsul corectat⟧' if has_c1 else
             '⟦the errors found in the AI answer of C2 and the corrected answer||erorile găsite în răspunsul AI din C2 și răspunsul corectat⟧')
    D.frame(_title(n, n, plural), items(
        ('⟦Outputs to produce||Rezultate de obținut⟧',
         ['⟦the answers to the A exercises||răspunsurile la exercițiile A⟧',
          '⟦the table or figure each B exercise asks for||tabelul sau figura cerute la fiecare exercițiu B⟧',
          c_out,
          '⟦the interpretation sentences each exercise asks for||frazele de interpretare cerute la fiecare exercițiu⟧']),
        not_graded,
        ('⟦Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}'
         '||Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}⟧',
         ['⟦each task names its notebook section||fiecare cerință indică secțiunea din notebook⟧']),
        CONTACT))
