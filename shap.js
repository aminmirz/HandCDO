/* Interactive SHAP figure: each row of the paper's beeswarm and each bar of the group chart is a target.
   Hover, focus, or tap a row to float a card with the parameter's meaning, its search range, and a clip of
   that parameter sweeping its range on a generated hand. Hovering a group bar highlights its rows. */
(() => {
  'use strict';
  const data = window.HandCDOShap; const plot = document.getElementById('shap-plot');
  if (!data || !plot) return;
  const card = document.getElementById('shap-card'), video = card.querySelector('video');
  const fine = matchMedia('(hover: hover) and (pointer: fine)');
  const { first, last, plotLeft, plotRight } = data.geometry, step = (last - first) / (data.rows.length - 1);

  const rows = data.rows.map((r, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'shap-row'; b.dataset.index = i;
    b.style.top = `${(first + i * step - step / 2) * 100}%`; b.style.height = `${step * 100}%`;
    b.style.left = `${plotLeft * 100}%`; b.style.width = `${(plotRight - plotLeft) * 100}%`;
    b.setAttribute('aria-label', `${r.label}: ${r.meaning} Range ${r.range}.`);
    b.setAttribute('aria-describedby', 'shap-card');
    plot.append(b); return b;
  });
  let current = -1, pinned = false, hideTimer = 0;
  function fill(i) {
    const r = data.rows[i];
    card.querySelector('.sc-rank').textContent = `#${r.rank} of ${data.rows.length}`;
    card.querySelector('.sc-label').textContent = r.label;
    card.querySelector('.sc-digit').innerHTML = `<i style="background:${r.digit.swatch}"></i>${r.digit.name} · ${r.digit.where}`;
    card.querySelector('.sc-meaning').textContent = r.meaning;
    card.querySelector('.sc-range').textContent = r.range;
    card.querySelector('.sc-group').textContent = r.group;
    if (video.dataset.id !== r.id) {
      video.dataset.id = r.id; video.poster = `media/params/${r.id}.webp`; video.src = `media/params/${r.id}.mp4`;
      video.play().catch(() => {});
    }
  }
  function place(i) {
    const wrap = card.parentElement.getBoundingClientRect(), pr = plot.getBoundingClientRect(), row = rows[i].getBoundingClientRect();
    if (!fine.matches || innerWidth < 900) { card.classList.add('docked'); card.style.left = card.style.top = ''; return; }
    card.classList.remove('docked');
    const w = card.offsetWidth, h = card.offsetHeight;
    let x = wrap.width + 18;                                // beside the plot, else over its right edge
    if (wrap.right + w + 24 > innerWidth) x = pr.left - wrap.left + pr.width * plotRight - w - 12;
    let y = row.top - wrap.top + row.height / 2 - h / 2;      // level with the row, kept on screen
    y = Math.max(-wrap.top + 80, Math.min(y, innerHeight - wrap.top - h - 16));
    card.style.left = `${x}px`; card.style.top = `${y}px`;
  }
  function show(i, pin = false) {
    clearTimeout(hideTimer);
    if (current !== i) { rows.forEach((b, k) => b.classList.toggle('on', k === i)); current = i; fill(i); }
    pinned = pin || pinned; card.hidden = false; card.classList.add('show'); place(i);
  }
  function hide(force = false) {
    if (pinned && !force) return;
    hideTimer = setTimeout(() => {
      card.classList.remove('show'); rows.forEach(b => b.classList.remove('on')); current = -1; pinned = false;
      video.pause();
    }, 120);
  }
  rows.forEach((b, i) => {
    b.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') show(i); });
    b.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') hide(); });
    b.addEventListener('focus', () => show(i));
    b.addEventListener('blur', () => hide());
    b.addEventListener('click', () => { if (pinned && current === i) { hide(true); } else { pinned = false; show(i, true); } });
    b.addEventListener('keydown', e => {
      const n = { ArrowDown: i + 1, ArrowUp: i - 1 }[e.key];
      if (n !== undefined && rows[n]) { e.preventDefault(); rows[n].focus(); }
      if (e.key === 'Escape') { hide(true); b.blur(); }
    });
  });
  card.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  card.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') hide(); });
  card.querySelector('.sc-close').addEventListener('click', () => hide(true));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(true); });
  addEventListener('resize', () => { if (current >= 0) place(current); });
  addEventListener('scroll', () => { if (current >= 0 && !card.classList.contains('docked')) place(current); }, { passive: true });

  document.addEventListener('handcdo:pause', () => video.pause());
})();
