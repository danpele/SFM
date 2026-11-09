"""Update the course-text keys of latex/web_text/chN_{ro,en}.md after deck changes that need no new prose.

Usage: python3 refresh_web_text_keys.py N [N ...]
- renames `## old title` to the current frame title when the builder log (web/chN_log.txt)
  reports `text key to rename: "## old" -> "## new"`;
- updates `{h=…}` to the current hash of every frame whose title is found in the deck.
Titles not found in the deck are left alone and reported (they need a manual decision).
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

for n in sys.argv[1:]:
    out = subprocess.run([sys.executable, 'build_web_lecture.py', n, '--hashes'], cwd=HERE, capture_output=True, text=True).stdout
    for lang in ('ro', 'en'):
        cur = {}
        for title, h in re.findall(rf'^{lang} p\. \d+: ## (.*?) \{{h=([0-9a-f]+)\}}$', out, re.M):
            cur.setdefault(title, h)
        renames = {}
        log = ROOT / 'web' / f'ch{n}_log.txt'          # one log for both languages
        if log.exists():
            for old, new in re.findall(r'text key to rename: (?:frame p\. \d+: )?"## (.*?)" -> "## (.*?)"', log.read_text(encoding='utf-8')):
                renames[old] = new
        path = HERE / 'web_text' / f'ch{n}_{lang}.md'
        src = path.read_text(encoding='utf-8')
        changed, renamed, missing = [], [], []

        def fix(m):
            title, old = m.group(1), m.group(2)
            if title not in cur and title in renames:
                renamed.append(title)
                title = renames[title]
            new = cur.get(title)
            if new is None:
                missing.append(title)
                return m.group(0)
            if new != old:
                changed.append(title)
            return f'## {title} {{h={new}}}'

        src = re.sub(r'^## (.*?) \{h=([0-9a-f]+)\}$', fix, src, flags=re.M)
        path.write_text(src, encoding='utf-8')
        print(f'ch{n} {lang}: {len(changed)} hashes updated, {len(renamed)} keys renamed'
              + (f'; NOT IN DECK: {missing}' if missing else ''))
