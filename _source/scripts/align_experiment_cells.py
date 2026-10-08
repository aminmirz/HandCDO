"""Find where each hand-camera cell of the experiments composite comes from in the raw camera clips.

experiments.mp4 (HandGeneration/experiments/plots) is a 3 x 3 grid (rows: stir, hammer, cut; columns: three hand
designs) beside animated slip plots. Each cell is a rotated, cropped and scaled window of one raw DSC_*.MOV clip.
For every cell this script recovers: source clip, crop rectangle (rotated raw pixels), and the time mapping
raw_t = speed * composite_t + offset, so high-resolution sections can be re-cut without changing the timing.

  python -B website_assets/scripts/align_experiment_cells.py stir|hammer|cut   # writes .runtime/exp-align/<task>.json
"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.runtime/exp-align'
HCD = Path.home() / 'Desktop/PhD/_Projects/HandCoDesign'
RAW = HCD / 'media/exp_vids'
COMP = HCD / 'HandGeneration/experiments/plots/experiments.mp4'
CW, CH = 881 / 3, 360.0                      # composite cell size (px)
TASKS = {  # row, candidate clips, coarse probe times, composite duration of the row (s)
    'stir': (0, ['DSC_0004', 'DSC_0018', 'DSC_0019', 'DSC_0049', 'DSC_0050', 'DSC_0051', 'DSC_0052'], [1.0, 4.0], 15.0),
    'hammer': (1, [f'DSC_{i:04d}' for i in range(26, 43)], [.6, 2.0], 3.0),
    'cut': (2, [f'DSC_{i:04d}' for i in list(range(20, 26)) + list(range(43, 49))], [1.0, 4.0], 12.3),
}
WIDTHS = [1080, 1000, 920, 840, 760, 680]
# composite time window in which each row's slip plot draws (measured from the plot column: plot_t = comp_t - start)
PLOT_WINDOW = {'stir': (0.0, 15.07), 'hammer': (2.41, 5.39), 'cut': (2.93, 15.07)}


def raw_frames(path, fps, w, h, start=None, dur=None, crop=None):
    vf = f'fps={fps},transpose=1'
    if crop: vf += ',crop={}:{}:{}:{}'.format(*crop)
    vf += f',scale={w}:{h}:flags=area,format=gray'
    cmd = ['ffmpeg', '-v', 'error'] + (['-ss', str(max(0, start))] if start is not None else []) + ['-i', str(path)]
    cmd += (['-t', str(dur)] if dur else []) + ['-vf', vf, '-f', 'rawvideo', '-']
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w).astype(np.float32)


def comp_cells(row, col, size, fps=None, t=None):
    """Composite cell(s) as gray arrays resized to size=(w, h); one frame at t, or the whole video at fps."""
    x0, y0, x1, y1 = round(col * CW) + 2, round(row * CH) + 2, round((col + 1) * CW) - 2, round((row + 1) * CH) - 2
    vf = f'crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={size[0]}:{size[1]}:flags=area,format=gray'
    cmd = ['ffmpeg', '-v', 'error'] + (['-ss', str(t)] if t is not None else []) + ['-i', str(COMP)]
    cmd += (['-frames:v', '1'] if t is not None else ['-r', str(fps)]) + ['-vf', vf, '-f', 'rawvideo', '-']
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, size[1], size[0]).astype(np.float32)


def ncc_search(stack, tpl):
    N, H, W = stack.shape; h, w = tpl.shape
    t = tpl - tpl.mean(); tn = np.sqrt((t * t).sum()) + 1e-6
    corr = np.fft.irfft2(np.fft.rfft2(stack, s=(H, W)) * np.conj(np.fft.rfft2(t, s=(H, W))), s=(H, W))[:, :H - h + 1, :W - w + 1]
    def box(a):
        c = np.cumsum(np.cumsum(np.pad(a, ((0, 0), (1, 0), (1, 0))), 1), 2)
        return c[:, h:, w:] - c[:, :-h, w:] - c[:, h:, :-w] + c[:, :-h, :-w]
    s1, s2 = box(stack), box(stack * stack)
    score = corr / (np.sqrt(np.maximum(s2 - s1 * s1 / (h * w), 1e-6)) * tn)
    flat = score.reshape(N, -1); i = flat.argmax(1)
    return flat.max(1), i, score.shape[1:]


def coarse(task):
    row, files, times, _ = TASKS[task]; F = 8; RW, RH = 1080 // F, 1920 // F
    cache = {f: raw_frames(RAW / f'{f}.MOV', 4, RW, RH) for f in files}
    found = {}
    for col in range(3):
        best = None
        for t in times:
            for cw in WIDTHS:
                w = round(cw / F); h = round(w * (CH - 4) / (CW - 4))
                if h > RH: continue
                tpl = comp_cells(row, col, (w, h), t=t)[0]
                for f, st in cache.items():
                    for k in range(0, len(st), 200):
                        m, idx, shp = ncc_search(st[k:k + 200], tpl); j = int(m.argmax())
                        if best is None or m[j] > best['score']:
                            y, x = divmod(int(idx[j]), shp[1])
                            best = dict(score=float(m[j]), file=f, raw_t=(k + j) / 4, comp_t=t, crop_w=cw, x=x * F, y=y * F)
        found[col] = best; print(task, col, best, flush=True)
    return found


def fine(task, col, c):
    """Refine the crop at 1/4 scale, then fit raw_t = a * comp_t + b over the whole 30 fps cell sequence."""
    row, _, _, dur = TASKS[task]; path = RAW / f"{c['file']}.MOV"
    F = 4; RW, RH = 1080 // F, 1920 // F
    win = raw_frames(path, 30, RW, RH, start=c['raw_t'] - 1, dur=2)
    best = None
    for cw in range(c['crop_w'] - 80, c['crop_w'] + 81, 20):
        if cw > 1080: continue
        w = round(cw / F); h = round(w * (CH - 4) / (CW - 4))
        tpl = comp_cells(row, col, (w, h), t=c['comp_t'])[0]
        m, idx, shp = ncc_search(win, tpl); j = int(m.argmax())
        if best is None or m[j] > best[0]:
            y, x = divmod(int(idx[j]), shp[1]); best = (float(m[j]), cw, x * F, y * F, w, h)
    score, cw, x, y, w, h = best
    ch = round(cw * (CH - 4) / (CW - 4)); crop = (cw, min(ch, 1920 - y), x, y)
    seq = comp_cells(row, col, (w, h), fps=30)                       # composite cell, every frame
    raw = raw_frames(path, 60, w, h, crop=crop)                       # raw crop, every frame at 60 fps
    def unit(a):
        a = a.reshape(len(a), -1); a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    S = unit(seq) @ unit(raw).T                                       # (composite frames, raw frames)
    live = np.where(seq.reshape(len(seq), -1).std(1) > 6)[0]          # skip blank (white) composite frames
    w0, w1 = PLOT_WINDOW[task]
    live = live[(live / 30.0 >= w0) & (live / 30.0 <= w1)]               # only while this row's plot is drawing
    tc = live / 30.0; best_fit = None
    for a in np.arange(.30, 3.01, .005):
        jj = np.round((a * tc[None, :] + np.arange(0, raw.shape[0] / 60, 1 / 60)[:, None]) * 60).astype(int)
        ok = (jj >= 0) & (jj < raw.shape[0])
        vals = np.where(ok, S[live[None, :].repeat(len(jj), 0), np.clip(jj, 0, raw.shape[0] - 1)], -1).mean(1)
        k = int(vals.argmax())
        if best_fit is None or vals[k] > best_fit[0]: best_fit = (float(vals[k]), float(a), k / 60)
    fit, a, b = best_fit
    per_frame = [float(S[i, int(round((a * i / 30 + b) * 60))]) if 0 <= round((a * i / 30 + b) * 60) < raw.shape[0] else None for i in live]
    return dict(file=c['file'], crop=dict(w=crop[0], h=crop[1], x=crop[2], y=crop[3]), speed=round(a, 4), offset_s=round(b, 4),
                mean_ncc=round(fit, 4), min_ncc=round(min(v for v in per_frame if v is not None), 4),
                live_comp_s=[round(float(tc[0]), 3), round(float(tc[-1]), 3)], coarse_ncc=c['score'])


def main():
    task = sys.argv[1]; WORK.mkdir(parents=True, exist_ok=True)
    cpath = WORK / f'{task}-coarse.json'
    found = {int(k): v for k, v in json.loads(cpath.read_text()).items()} if cpath.exists() else coarse(task)
    cpath.write_text(json.dumps(found, indent=2))
    out = {col: fine(task, col, c) for col, c in found.items()}
    (WORK / f'{task}.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
