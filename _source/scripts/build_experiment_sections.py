"""Cut the real-world experiment rows straight out of the presentation video, one video per task.

Source: HandCDO IROS Presentation/assets/exp_exported.mp4 (1920x1080, 30 fps), the edit used in the talk: a 3 x 3 grid
of camera cells (rows: stir, hammer, cut; columns: high, mid, low score hand) beside the animated slip plots. Each
section is that row's three cells exactly as in the talk (same pixels, same crop) over the window in which the row's
slip plot draws, so section time == plot time and the page's live plot stays in sync as in the original edit.

Plot windows were measured from the drawing tip of each plot: tip x advances at the axis scale (plot speed 1.0) from
composite t = 0.0 (stir), 2.41 (hammer) and 2.94 s (cut). Row/column seams: rows 0-401, 402-777, 778-1079; x 0-879.
Columns are re-ordered per row (ORDER) so that each shows the high, mid and low score hand in turn.
(The earlier raw-camera re-cut via align_experiment_cells.py is no longer used for the page.)

  python -B website_assets/scripts/build_experiment_sections.py
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / 'Desktop/PhD/_Projects/HandCoDesign/HandCDO IROS Presentation/assets/exp_exported.mp4'
OUT = ROOT / 'v2/media/experiments'
END = 13.63                                   # length of exp_exported.mp4 (s)
# task: (row y0, row height px, plot start in the composite s, plot window length s)
SECTIONS = {'stir': (0, 402, 0.0, END), 'hammer': (402, 376, 2.41, 2.98), 'cut': (778, 302, 2.94, END - 2.94)}
WIDTH = 880
CELLS = [(0, 292), (292, 293), (585, 295)]   # talk-video columns: x, width
# talk-video column shown as high, mid, low (the talk's cutting and hammering rows had hands out of order)
ORDER = {'stir': (0, 1, 2), 'hammer': (1, 0, 2), 'cut': (2, 0, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True); meta = {}
    for task, (y0, h, start, dur) in SECTIONS.items():
        dst = OUT / f'{task}.mp4'
        cells = ''.join(f'[0:v]crop={CELLS[c][1]}:{h}:{CELLS[c][0]}:{y0},setsar=1[c{k}];' for k, c in enumerate(ORDER[task]))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.3f}', '-i', str(SRC), '-t', f'{dur:.3f}',
                        '-filter_complex', cells + '[c0][c1][c2]hstack=inputs=3', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '0', '-i', str(dst), '-frames:v', '1', '-quality', '86', str(OUT / f'{task}.webp')], check=True)
        meta[task] = dict(duration=round(dur, 3), plot_start_in_composite=start, columns=['high', 'mid', 'low'],
                          source=SRC.name, row=dict(y=y0, h=h), talk_columns=list(ORDER[task]))
        print(task, dst.stat().st_size // 1024, 'KB', flush=True)
    (OUT / 'sections.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
