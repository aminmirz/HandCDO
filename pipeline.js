/* Interactive pipeline overview: hover, focus or tap a stage to lowlight the others and open a card about it.
   Numbers come from the paper and iros_paper_codebase/optimization/nested_optimizer_v2.py (default config). */
(() => {
  'use strict';
  const pipe = document.getElementById('pipe');
  if (!pipe) return;
  const C = 'media/components/', P = 'media/pipeline/';
  const STAGES = {
    A: { tag: 'A', title: 'Sample a design', group: 'Design optimization',
      text: 'A tree-structured Parzen estimator proposes the next hands, drawing new samples near past high scorers and away from low scorers.',
      facts: [['Sampler', 'TPE (Optuna)'], ['Search space', '28 selected design parameters'], ['Per iteration', '8 hands'], ['Grasp search', '64 iterations × 16 samples per hand and tool'], ['Converges', '≈ 200 iterations']],
      media: { img: C + 'res-progress.webp', alt: 'Grasp score rising over iterations' }, link: ['#step-a', 'How the loop works'] },
    B: { tag: 'B', title: 'Generate the hand', group: 'Design optimization',
      text: 'The parametric generator turns the sampled numbers into a palm, digits, fingertips and contact surfaces, and exports meshes and a URDF.',
      facts: [['Palm', 'outline sides, size, aspect ratio'], ['Digits', 'number, placement, angle, offsets'], ['Kinematics', 'finger / thumb codes, link lengths'], ['Contact', 'fingertip scale, surface kernels']],
      media: { video: P + 'general.mp4', poster: P + 'general.webp' }, link: ['#step-b', 'Explore the design space'] },
    T: { tag: 'Task', title: 'Demonstrated task grasp', group: 'Grasp optimization',
      text: 'For each tool, a person demonstrates the task with a power grasp while the tool’s 6-D pose is tracked. The clip replays the tracked hammering trajectory with a MANO hand.',
      facts: [['Tools', 'hammer, spoon (stir), knife'], ['Input', 'grasp pose + tool trajectory']],
      media: { video: 'media/clips/task.mp4', poster: 'media/clips/task.webp' }, link: ['#step-task', 'See this step'] },
    C: { tag: 'C', title: 'Grasp pose sampling', group: 'Grasp optimization',
      text: 'An inner search samples where the hand meets the tool, inside a region around the demonstrated grasp.',
      facts: [['Search', 'inner TPE per hand and tool'], ['Budget', '64 iterations × 16 samples (default)']],
      media: { video: P + 'graspsample.mp4', poster: P + 'graspsample.webp' }, link: ['#step-c', 'See this step'] },
    D: { tag: 'D', title: 'Joint-space sampling', group: 'Grasp optimization',
      text: 'For each grasp pose, joint configurations are sampled over the finger abduction joints and the first two thumb joints, with the other joints held open.',
      facts: [['Varies', 'finger abduction, thumb joints 1–2'], ['Within', 'each joint’s own limits']],
      media: { video: 'media/clips/close_front.mp4', poster: 'media/clips/close_front.webp' }, link: ['#step-de', 'See this step'] },
    E: { tag: 'E', title: 'Grasp initialization', group: 'Grasp optimization',
      text: 'The hand closes on the tool in physics simulation, starting from the sampled pose and joint configuration.',
      facts: [['Simulator', 'Isaac Lab / Isaac Sim'], ['Hands', 'batched, same topology per batch']],
      media: { video: 'media/clips/close_front.mp4', poster: 'media/clips/close_front.webp' }, link: ['#step-de', 'See this step'] },
    F: { tag: 'F', title: 'Wrench-space test score', group: 'Grasp optimization',
      text: 'The grasp is disturbed with forces and torques along and about X, Y and Z. Its resistance is the grasp score that goes back to the design optimizer.',
      facts: [['Disturbances', 'forces and torques on 3 axes'], ['Hand score', 'mean of the top-8 grasps']],
      media: { video: 'media/clips/wrench_front.mp4', poster: 'media/clips/wrench_front.webp' }, link: ['#step-f', 'Which parameters matter'] },
    G: { tag: 'Loop', title: 'Grasp optimization', group: 'Inner loop',
      text: 'For each generated hand and each tool, the inner loop iterates C → D → E → F: sample a grasp pose, sample joints, close the hand, test the grasp. The best grasps give the hand its score.',
      facts: [['Per hand', 'one search per tool'], ['Tools', 'hammer, spoon (stir), knife'], ['Hand score', 'mean of the top-8 grasps']],
      media: { video: 'media/overview/grasp.mp4', poster: P + 'grasp.webp' }, link: ['#framework', 'Walk through the loop'] },
  };
  const card = document.getElementById('pipe-card'), wrap = pipe.closest('.pipe-wrap');
  const stages = [...pipe.querySelectorAll('.ov-spot')], parts = [...pipe.querySelectorAll('.ov-layer, .ov-clip')];
  const fine = matchMedia('(hover: hover) and (pointer: fine)');
  let current = null, pinned = false, timer = 0;
  const SHOW_MEDIA = false;                                          // set true to show each stage's clip in the card

  function fill(id) {
    const s = STAGES[id];
    card.querySelector('.pc-tag').textContent = s.tag; card.querySelector('.pc-group').textContent = s.group;
    card.querySelector('.pc-title').textContent = s.title; card.querySelector('.pc-text').textContent = s.text;
    card.querySelector('.pc-facts').innerHTML = s.facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    const m = card.querySelector('.pc-media');                       // text only: the figure itself already animates
    if (SHOW_MEDIA) m.innerHTML = s.media.video ? `<video muted loop playsinline autoplay poster="${s.media.poster}" src="${s.media.video}"></video>`
      : `<img src="${s.media.img}" alt="${s.media.alt}">`;
    m.hidden = !SHOW_MEDIA; card.classList.toggle('no-media', !SHOW_MEDIA);
    const a = card.querySelector('.pc-link'); a.href = s.link[0]; a.textContent = `${s.link[1]} →`;
  }
  function place(el) {
    if (!fine.matches || innerWidth < 900) { card.classList.add('docked'); card.style.left = card.style.top = ''; return; }
    card.classList.remove('docked');
    const w = wrap.getBoundingClientRect(), r = el.getBoundingClientRect(), cw = card.offsetWidth, ch = card.offsetHeight;
    let x = r.left - w.left + r.width / 2 - cw / 2;
    x = Math.max(8, Math.min(x, w.width - cw - 8));
    let y = r.bottom - w.top + 12;                                   // below the stage, else above it
    if (r.bottom + ch + 24 > innerHeight && r.top - ch - 12 > 70) y = r.top - w.top - ch - 12;
    card.style.left = `${x}px`; card.style.top = `${y}px`;
  }
  function show(el, pin = false) {
    clearTimeout(timer); const id = el.dataset.stage;
    if (current !== id) {
      current = id; fill(id);
      stages.forEach(s => s.classList.toggle('on', s === el));
      parts.forEach(r => r.classList.toggle('on', r.dataset.stage === id));   // this stage's components stay, the rest fade
      pipe.classList.add('focus');
    }
    pinned = pin || pinned; card.hidden = false; card.classList.add('show'); place(el);
  }
  function hide(force = false) {
    if (pinned && !force) return;
    timer = setTimeout(() => {
      pipe.classList.remove('focus'); stages.forEach(s => s.classList.remove('on')); parts.forEach(r => r.classList.remove('on'));
      card.classList.remove('show'); current = null; pinned = false;
      const v = card.querySelector('video'); if (v) v.pause();
    }, 140);
  }
  stages.forEach(el => {
    el.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') show(el); });
    el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') hide(); });
    el.addEventListener('focus', () => show(el));
    el.addEventListener('blur', () => hide());
    el.addEventListener('click', () => { if (pinned && current === el.dataset.stage) hide(true); else { pinned = false; show(el, true); } });
    el.addEventListener('keydown', e => { if (e.key === 'Escape') { hide(true); el.blur(); } });
  });
  card.addEventListener('pointerenter', () => clearTimeout(timer));
  card.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') hide(); });
  card.querySelector('.pc-close').addEventListener('click', () => hide(true));
  addEventListener('resize', () => { const el = stages.find(s => s.dataset.stage === current); if (el) place(el); });

  // stage loops are lazy-loaded and played/paused by app.js like every other looping clip
})();
