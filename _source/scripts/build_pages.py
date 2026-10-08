"""Assemble the self-contained GitHub Pages site (gh-pages branch) from website_assets/v2.

  py -B website_assets/scripts/build_pages.py            -> website_assets/preview/pages/

The v2 page is served at the site root. Files it loads from outside v2/ (../site/media, ../site/vendor, ../figs,
../summary_video.mp4) are copied in under the same names without the '../'; only the copies are rewritten. The
'Classic page' footer link is dropped. The page sources and render scripts go in _source/ (MANO files excluded:
licence). Fails if a reference is left pointing outside the site or at a missing file.
"""
import re
import shutil
import sys
from pathlib import Path

WA = Path(__file__).resolve().parents[1]                  # website_assets/
V2, OUT = WA / 'v2', WA / 'preview/pages'
TEXT = {'.html', '.js', '.css'}
REWRITE = [('../site/', 'site/'), ('../figs/', 'figs/'), ('../summary_video.mp4', 'summary_video.mp4')]
DROP = ['<a href="../index.html">Classic page</a> &middot; ']
MAX_MB = 50
REF = re.compile(r'''(?:src|href|data-src|poster)\s*=\s*["']([^"'#?]+)|["'`]((?:\.\./|media/|models/|site/|figs/)[^"'`$?#]*)["'`]''')


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in V2.rglob('*'):                               # the page itself, at the root
        if f.is_file() and '__pycache__' not in f.parts: copy(f, OUT / f.relative_to(V2))
    outside = set()
    for f in OUT.rglob('*'):
        if f.suffix not in TEXT: continue
        s = f.read_text(encoding='utf-8')
        outside |= {m for m in re.findall(r'''\.\./[A-Za-z0-9_./-]+''', s)}
        for a in DROP: s = s.replace(a, '')
        for a, b in REWRITE: s = s.replace(a, b)
        f.write_text(s, encoding='utf-8')
    for ref in sorted(outside):                           # bring the outside files in
        if ref == '../index.html' or ref.endswith('/'): continue
        src = (V2 / ref).resolve()
        if src.is_file(): copy(src, OUT / ref[3:])
    copy(WA / 'site/vendor/three.min.js', OUT / 'site/vendor/three.min.js')
    for fig in (WA / 'figs').glob('*.jpg'): copy(fig, OUT / 'figs' / fig.name)
    (OUT / '.nojekyll').write_text('')
    # source, for rebuilding (not linked from the page)
    S = OUT / '_source'
    for name in ('README.md', '.gitignore'): copy(WA / name, S / name)
    for f in (WA / 'scripts').rglob('*'):
        rel = f.relative_to(WA / 'scripts')
        if not f.is_file() or '__pycache__' in rel.parts or rel.parts[:2] == ('data', 'mano'): continue
        copy(f, S / 'scripts' / rel)
    (S / 'README.md').write_text((WA / 'README.md').read_text(encoding='utf-8') + '\n\n## This folder\n'
        'Sources of the page served from this branch. The page itself is `v2/` in `website_assets/` (published here at the root '
        'by `scripts/build_pages.py`); the clips are rendered by the `scripts/build_*.py` / `render_*.py` scripts described above. '
        'MANO model files are not included (licence).\n', encoding='utf-8')
    # checks: nothing points outside, every local reference exists, no oversized file
    bad = []
    for f in OUT.rglob('*'):
        if f.suffix not in TEXT or '_source' in f.parts: continue
        s = f.read_text(encoding='utf-8')
        if '../' in s: bad.append(f'{f.relative_to(OUT)}: still contains ../')
        for a, b in REF.findall(s):
            r = a or b
            if not r or r.startswith(('http', 'mailto:', 'data:', '//', '#')) or '${' in r: continue
            if r.endswith('/') or '{' in r: continue
            if not (f.parent / r).exists() and not (OUT / r).exists(): bad.append(f'{f.relative_to(OUT)}: missing {r}')
    big = [f'{f.relative_to(OUT)} {f.stat().st_size / 2**20:.0f} MB' for f in OUT.rglob('*') if f.is_file() and f.stat().st_size > MAX_MB * 2**20]
    total = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) / 2**20
    n = sum(1 for f in OUT.rglob('*') if f.is_file())
    for b in bad + big: print('PROBLEM', b)
    print(f'{n} files, {total:.0f} MB -> {OUT}')
    sys.exit(1 if bad or big else 0)


if __name__ == '__main__':
    main()
