/* Page behaviour for the v2 page: lazy looping clips, framework beats, tabs, figure dialog, citation. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 980px)');
  let paused = reduced.matches;

  // ---------------------------------------------------------------- framework step A: TPE sampling on the slide's radar
  // Coordinates are the presentation slide's points (radar centre 115.0, 231.7; outer ring radius 64.9). Each iteration TPE draws
  // 8 candidates (n_samples_per_iter) from a density around the current good set plus some exploration, scores them on
  // a hidden objective, then re-splits every past sample into good (top 25%, green) and bad (red).
  const NS = 'http://www.w3.org/2000/svg', CX = 115.0, CY = 231.7, R = 60;   // visible radar ring r = 64.9 pt
  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  const OPT = [CX + 26, CY - 22];                                        // hidden optimum (illustrative)
  const score = p => Math.exp(-((p[0] - OPT[0]) ** 2 + (p[1] - OPT[1]) ** 2) / (2 * 24 ** 2)) + .06 * gauss();
  const inside = p => (p[0] - CX) ** 2 + (p[1] - CY) ** 2 < R * R;
  const uniform = () => { let p; do p = [CX + (rnd() * 2 - 1) * R, CY + (rnd() * 2 - 1) * R]; while (!inside(p)); return p; };
  let samples = [], iter = 0;
  function propose() {
    const good = samples.filter(s => s.cls === 'hi'), bw = Math.max(6, 20 - iter * 1.3);  // bandwidth shrinks as evidence grows
    if (iter < 2 || !good.length) return uniform();                        // startup trials are random
    for (let k = 0; k < 30; k++) {
      const g = good[Math.floor(rnd() * good.length)];
      const p = rnd() < .2 ? uniform() : [g.p[0] + bw * gauss(), g.p[1] + bw * gauss()];
      if (inside(p)) return p;
    }
    return uniform();
  }
  function split() {                                                         // TPE good/bad split by quantile
    const scored = [...samples].sort((a, b) => b.s - a.s);                 // every evaluated sample, newest included
    const n = Math.max(1, Math.ceil(scored.length * .25));
    scored.forEach((s, k) => { s.cls = k < n ? 'hi' : 'lo'; });
  }
  function draw() {
    document.querySelectorAll('svg.tpe').forEach(svg => {
      const g = svg.querySelector('.tpe-dots');
      while (g.childElementCount < samples.length) g.append(document.createElementNS(NS, 'circle'));
      while (g.childElementCount > samples.length) g.lastChild.remove();
      [...g.children].forEach((c, k) => {
        const s = samples[k]; c.setAttribute('cx', s.p[0].toFixed(1)); c.setAttribute('cy', s.p[1].toFixed(1)); c.setAttribute('r', '2.85');
        c.setAttribute('class', `dot ${s.cls}`); c.style.opacity = s.age > 3 ? Math.max(.25, 1 - (s.age - 3) * .15) : 1;
      });
      svg.querySelector('.tpe-iter').textContent = iter ? `iteration ${iter}` : '';
    });
  }
  function stepTPE() {
    if (iter >= 12) { samples = []; iter = 0; }                              // loop the illustration
    samples.forEach(s => { s.age++; }); samples = samples.filter(s => s.age < 8);
    iter++;
    for (let k = 0; k < 8; k++) { const p = propose(); samples.push({ p, s: score(p), cls: 'new', age: 0 }); }
    draw();
    setTimeout(() => { split(); draw(); }, 900);                            // evaluated in simulation, then re-split
  }
  const tpeVisible = () => document.querySelector('figure.is-active svg.tpe, .beat-media-inline svg.tpe');
  draw(); setInterval(() => { if (!paused && !document.hidden && tpeVisible()) stepTPE(); }, 2000);

  // ---------------------------------------------------------------- looping clips: load when near, play when visible
  const seen = new Map();
  const loops = () => [...document.querySelectorAll('video[data-src]')];
  function sync() {
    loops().forEach(v => {
      const show = seen.get(v) && !paused && !document.hidden && !v.closest('[hidden]') && !v.closest('.beat-frame figure:not(.is-active)');
      if (show) { if (!v.src) { v.src = v.dataset.src; } v.play().catch(() => {}); }
      else if (!v.paused) v.pause();
    });
  }
  const io = new IntersectionObserver(es => { es.forEach(e => seen.set(e.target, e.isIntersecting)); sync(); }, { threshold: .15 });
  const watch = v => { if (!v.dataset.watched) { v.dataset.watched = '1'; io.observe(v); } };
  loops().forEach(watch);
  document.addEventListener('visibilitychange', sync);

  const motionBtn = document.getElementById('motion-toggle');
  function setPaused(p) {
    paused = p; motionBtn.setAttribute('aria-pressed', String(p)); motionBtn.textContent = p ? 'Play clips' : 'Pause clips';
    if (p) document.dispatchEvent(new Event('handcdo:pause'));
    sync();
  }
  motionBtn.addEventListener('click', () => setPaused(!paused));
  reduced.addEventListener('change', () => setPaused(reduced.matches));
  setPaused(paused);

  // ---------------------------------------------------------------- design-space tabs
  const tabs = [...document.querySelectorAll('#ds-tabs [role="tab"]')];
  function pick(tab) {
    tabs.forEach(t => {
      const on = t === tab; t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    sync();
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => pick(t));
    t.addEventListener('keydown', e => {
      const n = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (n === undefined) return; e.preventDefault();
      const next = tabs[(n + tabs.length) % tabs.length]; next.focus(); pick(next);
    });
  });

  // ---------------------------------------------------------------- framework beats
  const beats = [...document.querySelectorAll('.beat')];
  const frames = [...document.querySelectorAll('#beat-frame figure')];
  const dots = [...document.querySelectorAll('#beat-progress button')];
  function activate(i) {
    beats.forEach((b, k) => b.classList.toggle('is-active', k === i));
    frames.forEach((f, k) => f.classList.toggle('is-active', k === i));
    dots.forEach((d, k) => { d.classList.toggle('on', k === i); d.setAttribute('aria-current', k === i ? 'step' : 'false'); });
    sync();
  }
  const beatIO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) activate(beats.indexOf(e.target)); }),
    { rootMargin: '-45% 0px -45% 0px' });
  beats.forEach(b => beatIO.observe(b));
  // On narrow screens the sticky stage is hidden; each beat gets its own copy of the media inline.
  function inlineMedia() {
    if (!narrow.matches) return;
    beats.forEach((b, i) => {
      const slot = b.querySelector('.beat-media-inline'); if (slot.childElementCount) return;
      const copy = frames[i].cloneNode(true); copy.className = 'card light';
      copy.querySelectorAll('[id]').forEach(e => e.removeAttribute('id'));
      copy.querySelectorAll('video').forEach(v => { v.removeAttribute('src'); delete v.dataset.watched; });
      slot.append(copy); copy.querySelectorAll('video[data-src]').forEach(watch);
    });
    sync();
  }
  narrow.addEventListener('change', inlineMedia); inlineMedia();
  activate(0);
  // jump to a stage from its title or the stage bar
  const go = i => beats[i].scrollIntoView({ block: 'center', behavior: reduced.matches ? 'auto' : 'smooth' });
  dots.forEach(d => d.addEventListener('click', () => go(+d.dataset.go)));
  beats.forEach((b, i) => b.querySelector('.beat-title').addEventListener('click', () => go(i)));

  // ---------------------------------------------------------------- reveal on scroll, nav highlight, score bars
  const rv = document.querySelectorAll('.pipe-wrap, .opt, .shap-plot, .exp, .abstract, .section-title, .section-sub, .clip, .demo, .bigstats > div, .design, .figure-thumb, .bib');
  const rvIO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); rvIO.unobserve(e.target); } }), { threshold: .1 });
  if (!reduced.matches) rv.forEach(el => { el.classList.add('rv'); rvIO.observe(el); });
  const designs = document.querySelector('.designs');
  new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { designs.classList.add('in'); o.disconnect(); } }, { threshold: .3 }).observe(designs);

  const navLinks = [...document.querySelectorAll('.topnav a:not(.brand)')];
  const navIO = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    navLinks.forEach(a => { const on = a.hash === `#${e.target.id}`; a.classList.toggle('active', on); on ? a.setAttribute('aria-current', 'location') : a.removeAttribute('aria-current'); });
  }), { rootMargin: '-40% 0px -55% 0px' });
  navLinks.forEach(a => { const s = document.querySelector(a.hash); if (s) navIO.observe(s); });

  // ---------------------------------------------------------------- figure dialog
  const dlg = document.getElementById('fig-dialog'), img = document.getElementById('fig-img');
  document.querySelectorAll('.figure-open').forEach(b => b.addEventListener('click', () => {
    img.src = `site/media/${b.dataset.figure}.webp`; img.alt = b.dataset.caption;
    document.getElementById('fig-caption').textContent = b.dataset.caption;
    document.getElementById('fig-full').href = `figs/${b.dataset.figure}.jpg`;
    dlg.showModal();
  }));
  document.getElementById('fig-close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  // ---------------------------------------------------------------- citation
  document.getElementById('copy-bib').addEventListener('click', async e => {
    const btn = e.currentTarget, text = document.getElementById('bib').textContent;
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch {
      const r = document.createRange(); r.selectNodeContents(document.getElementById('bib'));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    }
    btn.textContent = ok ? 'Copied' : 'Press Ctrl+C';
    document.getElementById('copy-status').textContent = ok ? 'BibTeX copied to clipboard.' : 'Citation selected; copy it with your keyboard.';
    setTimeout(() => { btn.textContent = 'Copy BibTeX'; }, 2200);
  });
})();
