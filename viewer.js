/* Live 3D stage for the v2 page. Shows the exact generation_v2 exports in site/models; never synthesizes geometry.
   Each model.js calls window.HandCDOModels.register(data) with base64 meshes, a node tree, and revolute joints. */
(() => {
  'use strict';
  const T = window.THREE;
  // Compact and non-anthropomorphic designs exported by scripts/build_creative_hands.py (generation_v2, full outline placement).
  // pitch: default viewing elevation; hands stand upright with the palm facing the camera.
  const MODELS = [
    { id: 'compact', label: 'Compact', digits: 4, pitch: .14 },
    { id: 'duo', label: 'Two + thumb', digits: 3, pitch: .14 },
    { id: 'side-four', label: 'Side finger', digits: 5, pitch: .14 },
    { id: 'radial-four', label: 'Cross', digits: 4, pitch: .14 },
    { id: 'tri-claw', label: 'Tri-claw', digits: 3, pitch: .14 },
    { id: 'opposed', label: 'Opposed', digits: 4, pitch: .14 },
    { id: 'pinch-plus', label: '3 + 2', digits: 5, pitch: .14 },
    { id: 'star-five', label: 'Star', digits: 5, pitch: .14 },
    { id: 'long-reach', label: 'Long reach', digits: 4, pitch: .14 },
  ];
  const base = id => `models/${id}`;
  const host = document.getElementById('hand-viewer');
  if (!host) return;
  const loading = document.getElementById('stage-loading'), presets = document.getElementById('presets');
  const resetBtn = document.getElementById('reset-view'), rotateBtn = document.getElementById('auto-rotate');
  const graspBtn = document.getElementById('grasp-demo');
  const flexIn = document.getElementById('flex'), spreadIn = document.getElementById('spread');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  // ---------------------------------------------------------------- model loading (script tags work from file:// too)
  const cache = new Map(), pending = new Map();
  window.HandCDOModels = { register(data) {
    if (data.schema !== 'handcdo-generation-v2-mesh/1') return;
    cache.set(data.id, data);
    const p = pending.get(data.id); if (p) { p.resolve(data); pending.delete(data.id); }
  } };
  function load(id) {
    if (cache.has(id)) return Promise.resolve(cache.get(id));
    if (pending.has(id)) return pending.get(id).promise;
    let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; });
    pending.set(id, { resolve, reject, promise });
    const s = document.createElement('script'); s.src = `${base(id)}/model.js`; s.async = true;
    s.onerror = () => { pending.delete(id); s.remove(); reject(new Error('This hand could not be loaded.')); };
    s.onload = () => { s.remove(); if (!cache.has(id)) { pending.delete(id); reject(new Error('The model file is invalid.')); } };
    document.head.append(s); return promise;
  }

  // ---------------------------------------------------------------- preset buttons
  MODELS.forEach((m, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'radio'); b.dataset.id = m.id;
    b.setAttribute('aria-checked', String(i === 0)); b.tabIndex = i === 0 ? 0 : -1;
    b.innerHTML = `<span class="dots" aria-hidden="true">${'<i></i>'.repeat(m.digits)}</span>${m.label}`;
    b.addEventListener('click', () => select(m.id));
    b.addEventListener('keydown', e => {
      const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!k) return;
      e.preventDefault(); const n = MODELS[(i + k + MODELS.length) % MODELS.length];
      presets.querySelector(`[data-id="${n.id}"]`).focus(); select(n.id);
    });
    presets.append(b);
  });

  function fail(msg) {
    loading.textContent = msg; loading.classList.remove('done'); loading.classList.add('error');
  }
  if (!T) { document.getElementById('stage-fallback').hidden = false; fail('3D is unavailable in this browser.'); return; }

  // ---------------------------------------------------------------- renderer and studio
  let renderer;
  try {
    renderer = new T.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  } catch (e) { document.getElementById('stage-fallback').hidden = false; fail('3D is unavailable in this browser.'); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = T.SRGBColorSpace;
  renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.12;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  host.append(renderer.domElement);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(30, 1, .01, 100);
  scene.add(new T.HemisphereLight(0xffffff, 0x4a4643, 1.6));
  const key = new T.DirectionalLight(0xfff4ea, 2.6); key.position.set(-3, 6, 5); scene.add(key);
  const rim = new T.DirectionalLight(0xdfe9ff, 1.4); rim.position.set(4, 2.5, -5); scene.add(rim);
  const fill = new T.DirectionalLight(0xffffff, .55); fill.position.set(2, -3, 4); scene.add(fill);

  // soft contact shadow under the hand
  const shadowTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
    const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    r.addColorStop(0, 'rgba(0,0,0,.55)'); r.addColorStop(.55, 'rgba(0,0,0,.18)'); r.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = r; g.fillRect(0, 0, 128, 128); return new T.CanvasTexture(c);
  })();
  const shadow = new T.Mesh(new T.PlaneGeometry(1, 1), new T.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; scene.add(shadow);

  // ---------------------------------------------------------------- scene state
  let display = null, geoms = [], mats = [], joints = [], current = null, version = 0;
  let yaw = -.32, pitch = MODELS[0].pitch, dist = 1, tYaw = yaw, tPitch = pitch, tDist = dist;
  const HOME = { yaw: -.32, pitch: MODELS[0].pitch, dist: 1 };
  let autoRotate = !reduced.matches, demo = false, demoT = 0, flex = 0, spread = 0, flexShown = 0, spreadShown = 0;
  let pop = 1, visible = false, raf = 0, last = 0, userMoved = false, dragging = false;
  rotateBtn.setAttribute('aria-pressed', String(autoRotate));

  const b64 = (s, Type) => { const raw = atob(s), u = new Uint8Array(raw.length); for (let i = 0; i < raw.length; i++) u[i] = raw.charCodeAt(i); return new Type(u.buffer); };

  function build(data) {
    const nextGeoms = data.geometries.map(src => {
      const g = new T.BufferGeometry();
      g.setAttribute('position', new T.BufferAttribute(b64(src.positions, Float32Array), 3));
      g.setIndex(new T.BufferAttribute(b64(src.indices, Uint32Array), 1));
      src.groups.forEach(gr => g.addGroup(...gr));
      g.computeVertexNormals(); g.computeBoundingSphere(); return g;
    });
    const nextMats = data.materials.map(m => new T.MeshStandardMaterial({
      name: m.name, color: new T.Color().setRGB(...m.color), roughness: Math.min(.85, m.roughness + .1),
      metalness: m.metalness, opacity: m.opacity, transparent: m.opacity < 1, flatShading: true,
    }));
    const root = new T.Group(), inner = new T.Group(); root.add(inner);
    const objs = data.nodes.map(n => {
      const o = n.geometry === undefined ? new T.Group() : new T.Mesh(nextGeoms[n.geometry], n.materials.map(i => nextMats[i]));
      o.name = n.name; o.matrix.fromArray(n.matrix); o.matrixAutoUpdate = false; return o;
    });
    data.nodes.forEach((n, i) => (n.parent >= 0 ? objs[n.parent] : inner).add(objs[i]));
    // generator frame used as is: fingers along +Y (up), palm / grasp side along +Z (towards the default camera)
    root.updateMatrixWorld(true);
    const box = new T.Box3().setFromObject(root), size = box.getSize(new T.Vector3()), center = box.getCenter(new T.Vector3());
    const s = 1 / Math.max(size.x, size.y, size.z);
    root.scale.setScalar(s); root.userData.scale = s; root.userData.radius = size.length() / 2 * s; root.position.set(-center.x * s, -box.min.y * s - .5 * size.y * s, -center.z * s);
    shadow.position.y = -.5 * size.y * s - .004; shadow.scale.setScalar(1.25 * Math.max(size.x, size.z) * s);

    // classify joints: symmetric small ranges act as spread (abduction), others flex toward their larger-magnitude limit
    const chains = [...new Set(data.joints.map(j => j.chain))].filter(c => c.startsWith('finger'));
    const nextJoints = data.joints.map(j => {
      const o = objs[j.node], sym = Math.abs(j.lower + j.upper) < .05;
      const ci = chains.indexOf(j.chain), side = chains.length > 1 ? (ci / (chains.length - 1)) * 2 - 1 : 0;
      return { ...j, o, rest: o.matrix.clone(), ax: new T.Vector3(...j.axis).normalize(), sym, side,
        target: Math.abs(j.upper) >= Math.abs(j.lower) ? j.upper : j.lower };
    });
    return { root, nextGeoms, nextMats, nextJoints };
  }

  const tmp = new T.Matrix4();
  // Flex 100% = each digit's collision-checked closing limit; Spread +/-100% = the limit in that direction
  // (fractions computed at export by scripts/motion_limits.py and stored in model parameters.motion).
  let motion = null;
  function pose() {
    const sp = spreadShown < 0 ? spreadShown * (motion ? motion.spread[0] : 1) : spreadShown * (motion ? motion.spread[1] : 1);
    joints.forEach(j => {
      const reach = motion ? (motion.flex[j.chain] ?? 0) : .85;
      let a = j.sym ? sp * j.side * j.upper : flexShown * reach * j.target;
      a = Math.max(j.lower, Math.min(j.upper, a));
      j.o.matrix.copy(j.rest).multiply(tmp.makeRotationAxis(j.ax, a)); j.o.matrixWorldNeedsUpdate = true;
    });
  }

  async function select(id) {
    const v = ++version, m = MODELS.find(x => x.id === id);
    presets.querySelectorAll('button').forEach(b => { const on = b.dataset.id === id; b.setAttribute('aria-checked', String(on)); b.tabIndex = on ? 0 : -1; });
    loading.textContent = `Loading ${m.label.toLowerCase()}…`; loading.classList.remove('done', 'error');
    try {
      const data = await load(id); if (v !== version) return;
      const { root, nextGeoms, nextMats, nextJoints } = build(data);
      if (display) scene.remove(display);
      geoms.forEach(g => g.dispose()); mats.forEach(x => x.dispose());
      display = root; geoms = nextGeoms; mats = nextMats; joints = nextJoints; current = data;
      motion = data.parameters.motion || null;
      const [neg, pos] = motion ? motion.spread : [1, 1];       // only directions with room to move
      spreadIn.min = neg > 0 ? -100 : 0; spreadIn.max = pos > 0 ? 100 : 0; spreadIn.disabled = !(neg > 0 || pos > 0);
      spread = spreadShown = 0; spreadIn.value = 0; out('spread', 0);
      scene.add(display); pop = reduced.matches ? 1 : 0; pose();
      HOME.pitch = m.pitch; if (!userMoved) tPitch = m.pitch;
      loading.classList.add('done');
      const p = data.parameters;
      setText('stat-digits', `${p.finger_number} + ${p.thumb_number}`);
      setText('stat-joints', data.stats.joints);
      setText('stat-palm', `${Math.round(p.palm_size_mm)} mm`);
      setText('stat-tris', `${(data.stats.triangles / 1e6).toFixed(2)} M`);
      const dl = document.getElementById('config-download'); if (dl) dl.href = `${base(id)}/configuration.zip`;
      host.setAttribute('aria-label', `Generated hand, ${m.label}: ${p.finger_number} fingers and ${p.thumb_number} thumb, ${data.stats.joints} joints. Drag or use arrow keys to turn; plus and minus zoom; Home resets the view.`);
      wake();
    } catch (e) { if (v === version) fail(e.message); }
  }
  const setText = (id, t) => { const el = document.getElementById(id); if (el) el.textContent = t; };

  // ---------------------------------------------------------------- loop
  function frame(time) {
    raf = 0; if (!visible || document.hidden) return;
    const dt = Math.min(.05, (time - last) / 1000 || 0); last = time;
    const k = 1 - Math.exp(-dt * 9);
    if (autoRotate && !dragging) tYaw += dt * .28;
    if (demo) { demoT += dt; flex = .5 - .5 * Math.cos(demoT * 1.6); flexIn.value = Math.round(flex * 100); out('flex', flex); }
    yaw += (tYaw - yaw) * k; pitch += (tPitch - pitch) * k; dist += (tDist - dist) * k;
    flexShown += (flex - flexShown) * (reduced.matches ? 1 : k); spreadShown += (spread - spreadShown) * (reduced.matches ? 1 : k);
    if (display) { pop += (1 - pop) * (1 - Math.exp(-dt * 7)); display.scale.setScalar(display.userData.scale * (.9 + .1 * pop)); pose(); }
    place(); renderer.render(scene, camera);
    const settling = Math.abs(tYaw - yaw) + Math.abs(tPitch - pitch) + Math.abs(tDist - dist) + Math.abs(flex - flexShown) + Math.abs(spread - spreadShown) + (1 - pop) > 1e-4;
    if (autoRotate || demo || settling || dragging) raf = requestAnimationFrame(frame);
  }
  function place() {
    const aspect = Math.max(.4, host.clientWidth / Math.max(1, host.clientHeight));
    // fit the hand's bounding sphere, whatever its layout (tall anthropomorphic or flat radial)
    const r = display ? display.userData.radius : .7;
    const d = 1.25 * r / Math.sin(T.MathUtils.degToRad(camera.fov / 2)) * dist * Math.max(1, 1.05 / aspect);
    camera.position.set(Math.sin(yaw) * Math.cos(pitch) * d, Math.sin(pitch) * d + .03, Math.cos(yaw) * Math.cos(pitch) * d);
    camera.lookAt(0, .03, 0);
  }
  function wake() { if (!raf && visible) { last = performance.now(); raf = requestAnimationFrame(frame); } }
  function resize() {
    const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h;
    camera.setViewOffset(w, h, 0, Math.min(90, h * .1), w, h);   // lift the hand clear of the control dock
    camera.updateProjectionMatrix(); wake();
  }
  new ResizeObserver(resize).observe(host);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; wake(); }, { rootMargin: '80px' }).observe(host);
  document.addEventListener('visibilitychange', wake);

  // ---------------------------------------------------------------- interaction
  function moved() { if (!userMoved) { userMoved = true; resetBtn.classList.add('show'); } }
  function home() { tYaw = HOME.yaw + Math.round((yaw - HOME.yaw) / (2 * Math.PI)) * 2 * Math.PI; tPitch = HOME.pitch; tDist = HOME.dist; userMoved = false; resetBtn.classList.remove('show'); wake(); }
  function setAuto(on) { autoRotate = on; rotateBtn.setAttribute('aria-pressed', String(on)); wake(); }
  let px = 0, py = 0, pid = null;
  host.addEventListener('pointerdown', e => {
    if (e.button !== 0) return; dragging = true; pid = e.pointerId; px = e.clientX; py = e.clientY;
    host.setPointerCapture(pid); setAuto(false); wake();
  });
  host.addEventListener('pointermove', e => {
    if (!dragging || e.pointerId !== pid) return;
    tYaw -= (e.clientX - px) * .0085; tPitch = Math.max(-1.2, Math.min(1.35, tPitch + (e.clientY - py) * .006));
    px = e.clientX; py = e.clientY; moved(); wake();
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(t => host.addEventListener(t, () => { dragging = false; }));
  host.addEventListener('dblclick', home);
  // zoom only with a modifier or pinch (ctrlKey on trackpads) so the page keeps scrolling
  host.addEventListener('wheel', e => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault(); tDist = Math.max(.45, Math.min(1.8, tDist * Math.exp(e.deltaY * .0025))); moved(); wake();
  }, { passive: false });
  host.addEventListener('keydown', e => {
    const step = { ArrowLeft: [-.15, 0], ArrowRight: [.15, 0], ArrowUp: [0, .1], ArrowDown: [0, -.1] }[e.key];
    if (step) { tYaw += step[0]; tPitch = Math.max(-1.2, Math.min(1.35, tPitch + step[1])); setAuto(false); moved(); }
    else if (e.key === '+' || e.key === '=') { tDist = Math.max(.45, tDist * .88); moved(); }
    else if (e.key === '-') { tDist = Math.min(1.8, tDist / .88); moved(); }
    else if (e.key === 'Home') home();
    else return;
    e.preventDefault(); wake();
  });
  resetBtn.addEventListener('click', home);
  rotateBtn.addEventListener('click', () => setAuto(!autoRotate));
  const out = (name, v) => { document.getElementById(`${name}-out`).textContent = `${Math.round(v * 100)}%`; };
  function setDemo(on) { demo = on; graspBtn.setAttribute('aria-pressed', String(on)); if (on) demoT = Math.acos(1 - 2 * flex) / 1.6; wake(); }
  flexIn.addEventListener('input', () => { setDemo(false); flex = flexIn.value / 100; out('flex', flex); wake(); });
  spreadIn.addEventListener('input', () => { spread = spreadIn.value / 100; out('spread', spread); wake(); });
  graspBtn.addEventListener('click', () => setDemo(!demo));
  reduced.addEventListener('change', () => { if (reduced.matches) { setAuto(false); setDemo(false); } });
  document.addEventListener('handcdo:pause', () => { setAuto(false); setDemo(false); });

  renderer.domElement.addEventListener('webglcontextlost', e => {
    e.preventDefault(); renderer.domElement.hidden = true; document.getElementById('stage-fallback').hidden = false;
    fail('The 3D view was lost. Reload the page to restore it.');
  });

  resize(); select(MODELS[0].id);
})();
