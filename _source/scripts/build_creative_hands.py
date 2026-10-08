"""Export the v2 3D-stage designs (creative_hands.DESIGNS) to website_assets/v2/models/<id>/.

  python -B website_assets/scripts/build_creative_hands.py [--ids compact star-five ...] [--sketch out.png]
"""
import sys
sys.dont_write_bytecode = True
import argparse, os, subprocess
from pathlib import Path
from creative_hands import DESIGNS, ROOT, sketch

BLENDER = os.environ.get('HANDCDO_BLENDER', 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ids', nargs='+', choices=list(DESIGNS), default=list(DESIGNS))
    ap.add_argument('--sketch', help='only draw a top-view layout sketch to this PNG'); args = ap.parse_args()
    if args.sketch:
        sketch(args.ids, args.sketch); return
    temp = ROOT / '.runtime/blender-temp'; temp.mkdir(parents=True, exist_ok=True)
    log = ROOT / 'preview/creative.log'; log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run([BLENDER, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                                 str(ROOT / 'scripts/export_creative_hands.py'), '--', '--ids', *args.ids],
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TEMP=str(temp), TMP=str(temp)),
                                stdout=out, stderr=subprocess.STDOUT)
    lines = [l for l in log.read_text(encoding='utf-8', errors='replace').splitlines() if 'CREATIVE_EXPORT' in l]
    print('\n'.join(lines))
    if result.returncode:
        raise SystemExit(f'Blender failed; see {log}')


if __name__ == '__main__':
    main()
