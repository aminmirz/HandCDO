/* Real-world experiment rows: each task's three hand cameras (one high-resolution video) beside a live slip plot.
   Curves are the averaged in-hand rotational slip from the OptiTrack data (media/experiments/slip.js); the plot is
   drawn up to the video's current time, which equals plot time (sections are cut to the plot window of the original
   edit). Hovering a camera column or a legend entry highlights that hand's curve. */
(() => {
  'use strict';
  const D = window.HandCDOSlip && window.HandCDOSlip.data;
  const rows = [...document.querySelectorAll('.xp')];
  if (!D || !rows.length) return;
  const NS = 'http://www.w3.org/2000/svg';
  const COLORS = { high: '#32cd32', mid: '#4169e1', low: '#ff4500' };            // limegreen / royalblue / orangered, as in the paper
  const TICKS = { spoon: 2, hammer: .5, knife: 2 };
  const VW = innerWidth < 700 ? 420 : 640, VH = 250, M = { l: 44, r: 12, t: 12, b: 30 };   // narrower plot on phones keeps the labels readable
  const el = (n, a, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (p) p.append(e); return e; };

  rows.forEach(row => {
    const tool = row.dataset.tool, d = D[tool], video = row.querySelector('video'), svg = row.querySelector('svg.xp-plot');
    const x0 = 0, x1 = d.x_max, y0 = Math.max(0, Math.floor(d.y_lo / 10) * 10), y1 = Math.ceil(d.y_hi / 10) * 10;
    const X = t => M.l + (t - x0) / (x1 - x0) * (VW - M.l - M.r), Y = v => VH - M.b - (v - y0) / (y1 - y0) * (VH - M.t - M.b);
    svg.setAttribute('viewBox', `0 0 ${VW} ${VH}`);
    const grid = el('g', { class: 'xp-grid' }, svg);
    for (let t = 0; t <= x1 + 1e-6; t += TICKS[tool]) {
      el('line', { x1: X(t), x2: X(t), y1: M.t, y2: VH - M.b }, grid);
      el('text', { x: X(t), y: VH - M.b + 18, 'text-anchor': 'middle' }, grid).textContent = TICKS[tool] < 1 ? t.toFixed(1) : t;
    }
    const ystep = y1 - y0 > 80 ? 50 : 20;
    for (let v = Math.ceil(y0 / ystep) * ystep; v <= y1; v += ystep) {
      el('line', { x1: M.l, x2: VW - M.r, y1: Y(v), y2: Y(v) }, grid);
      el('text', { x: M.l - 7, y: Y(v) + 4, 'text-anchor': 'end' }, grid).textContent = v;
    }
    el('rect', { x: M.l, y: M.t, width: VW - M.l - M.r, height: VH - M.t - M.b, class: 'xp-frame' }, svg);
    el('text', { x: 12, y: (VH - M.b + M.t) / 2, class: 'xp-ylab', transform: `rotate(-90 12 ${(VH - M.b + M.t) / 2})` }, svg).textContent = 'Slip (deg)';
    const curves = {};
    ['high', 'mid', 'low'].forEach(k => {
      const h = d.hands[k]; if (!h) return;
      const g = el('g', { class: `xp-hand ${k}`, 'data-hand': k }, svg);
      el('line', { x1: M.l, x2: VW - M.r, y1: Y(h.mean), y2: Y(h.mean), stroke: COLORS[k], class: 'xp-mean' }, g);
      const line = el('polyline', { stroke: COLORS[k], class: 'xp-line' }, g);
      curves[k] = { g, line, pts: h.t.map((t, i) => [X(t), Y(h.v[i]), t]) };
    });
    let last = -1;
    function draw() {
      const now = video.currentTime || 0;
      if (Math.abs(now - last) > 1e-3) {
        last = now;
        for (const k in curves) {
          const c = curves[k]; let n = 0; while (n < c.pts.length && c.pts[n][2] <= now) n++;
          c.line.setAttribute('points', c.pts.slice(0, n).map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
        }
      }
      raf = visible ? requestAnimationFrame(draw) : 0;
    }
    let raf = 0, visible = false;
    new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && !raf) raf = requestAnimationFrame(draw); }, { threshold: .05 }).observe(row);
    // hover / focus: highlight one hand everywhere in this row
    const focus = k => {
      row.classList.toggle('focus', !!k);
      row.querySelectorAll('[data-hand]').forEach(e => e.classList.toggle('on', e.dataset.hand === k));
    };
    row.querySelectorAll('.xp-col, .xp-legend button').forEach(b => {
      b.addEventListener('pointerenter', () => focus(b.dataset.hand)); b.addEventListener('pointerleave', () => focus(null));
      b.addEventListener('focus', () => focus(b.dataset.hand)); b.addEventListener('blur', () => focus(null));
      b.addEventListener('click', () => focus(row.querySelector('.on') && row.querySelector('.on').dataset.hand === b.dataset.hand ? null : b.dataset.hand));
    });
  });
})();
