# HandCDO project website

The redesigned standalone entry point is `website_assets/index.html`. All redesign code, media derivatives, and scripts live in this directory. The repository-root `index.html` is unchanged. The page uses a white academic layout with a centered paper title, authors, resource links, research sections, and subtle scroll transitions. Reduced-motion preferences and the animation pause control are respected.

## View the original hand models

Open `website_assets/index.html` directly in a modern browser, or serve this folder from the repository root:

```powershell
conda run -n base python -m http.server 8080 --directory website_assets
```

Then open http://localhost:8080. The static site includes seven configurations exported from the original `generation_v2/blender/HandGeneratorV2.py` add-on and `components.blend`. It does not reconstruct the hand with approximate browser geometry. No npm installation, external fonts, or runtime CDN access is needed. Publishing at the repository root would require a separate change to its existing entry point.

The browser preserves the exported vertices, triangles, assembly transforms, material parameters, joint axes, and joint limits. Only the camera, lighting, and uniform display scale are supplied by the viewer. Joint motion is a kinematic demonstration, not a simulated grasp or a collision-checked trajectory.

## Generate custom hands locally

From the repository root, run:

```powershell
conda run --no-capture-output -n base python -B website_assets/scripts/serve_generator.py
```

Open **http://127.0.0.1:8080**. This enables parameter editing. Select **Generate hand** to call the actual V2 Blender add-on, wait for generation, and load the resulting meshes. Pending edits leave the previously generated hand visible until a new build succeeds. Errors are reported without substituting approximate geometry. The service listens only on loopback; stop it with Ctrl+C.

The static webpage supports preset selection, original joint articulation, and configuration downloads. Arbitrary new configurations require the local service above, or a separately deployed Blender backend. A static hosting service cannot execute Blender by itself.

The controls map to native properties:

| Website control | Original add-on property |
| --- | --- |
| Finger count | `finger_number` (one right-side thumb) |
| Palm size | `palm_size_mm` |
| Palm polygon sides | `outline_sides` |
| Base angle spread | Symmetric per-digit `finger_i_angle` |
| Added link length | Per-digit `finger_i_link_added_length`, in mm |
| Fingertip scale | Per-digit `finger_i_fingertip_scale` and `thumb_0_fingertip_scale` |
| Pad deformation | Gaussian `pad_max_intensity` and per-finger pad intensity, capped at 4 mm for fingers |

The V2 default finger code is `2-1-1`; the thumb code is `1--22`. Remaining parameters use the add-on defaults, except explicit website configuration values recorded in each export. Collision-mesh generation is disabled for these visualization exports; it does not alter the visible component meshes.

### Dependencies

Tested with Blender 5.2.2 and its Python 3.13. The source add-on targets Blender 4.5 or later. Set `HANDCDO_BLENDER` or pass `--blender` to select an executable. The server uses Python's standard library. Blender needs NumPy (bundled), SciPy, and pyclipper. The latter two have been installed locally in `website_assets/.runtime/python`; no changes to `generation_v2` are required.

For a fresh checkout, install into this folder using the Python executable bundled with **your** Blender version. For this machine:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe' -m pip install --no-cache-dir --no-deps --target website_assets/.runtime/python scipy==1.18.1 pyclipper==1.4.0
```

The export wrapper prioritizes these dependencies over user-installed Blender modules and disables Python bytecode writes. The add-on's temporary files, configs, generated models, and logs stay under `website_assets`.

### Rebuild the preset library

```powershell
conda run --no-capture-output -n base python -B website_assets/scripts/build_v2_library.py --save-blend
```

Use `--presets baseline surface` to rebuild selected configurations. The generator source and component files are read in place, not copied or modified. Each exported model includes SHA-256 source hashes for provenance. Rebuild after updating `generation_v2`; restart the local server after source changes so its cache version is refreshed.

**Export displayed model settings** downloads the actual add-on property values as JSON. **Download native configuration files** contains the original `palm_cfg.py`, `finger_cfg_*.py`, `thumb_cfg_*.py`, `hand_cfg.py`, and assembly NPZ files, plus metadata. These describe the displayed model, even if form edits are still pending. To regenerate from an exported settings JSON:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python website_assets/scripts/export_v2_model.py -- --id reproduced --parameters path/to/handcdo-v2-addon-settings.json --save-blend
```

## Design and interactions

- White research-project layout with the full paper title, abstract, technical section headings, paper/code/video links, and the original author information.
- Four keyboard-accessible component tabs with the existing Blender animations.
- Original V2 hand meshes with orbit, zoom, source joint articulation, and seven generated configurations.
- Local Blender-backed generation and native configuration exports. The viewer does not run grasp optimization or calculate performance scores.
- Figure dialogs link to the full-resolution source images. Escape closes the dialog.
- Existing experimental video and reported scores; no synthetic performance values.
- Copyable arXiv BibTeX, local dependencies, reduced-motion support, and a static fallback for browsers without WebGL.
- Offscreen video and rendering pause to reduce resource use. Orbit is opt-in. Keyboard camera controls: arrows, `+`, `-`, and `Home`.

## Files

- `site/style.css`, `site/app.js`: page styling and interactions.
- `site/explorer.js`: original-mesh loader, joint articulation, and local-generation interface.
- `site/models/`: seven original V2 model exports, native configurations, source metadata, and optional Blender snapshots. Each `model.js` loads on selection; shared component geometry is deduplicated without changing vertices or triangles.
- `site/generator-config.js`: static-mode default; the local server supplies its service configuration at this URL.
- `scripts/export_v2_model.py`: Blender wrapper around the unchanged source add-on.
- `scripts/generator_config.py`: explicit website-to-add-on parameter mapping.
- `scripts/build_v2_library.py`: reproducible preset generation and native config packaging.
- `scripts/serve_generator.py`: local HTTP server and validated, single-job Blender generation API.
- `.runtime/`: ignored dependencies, temporary Blender data, and custom-model cache.
- `site/vendor/three.min.js`: Three.js 0.160.1, MIT licensed.
- `site/media/`: optimized WebP derivatives and a short compressed hero video.
- `assets/`, `figs/`, `summary_video.mp4`: original project media, preserved.
- `scripts/prepare_media.py`: regenerate image derivatives (Pillow and FFmpeg required).
- `scripts/check_site.py`: browser checks (Python Playwright and Chrome required). Set `HANDCDO_CHROME` to override the Chrome executable. Temporary browser data, screenshots, and the check report stay in `preview/`.

Run checks from the repository root:

```powershell
conda run -n base python -B website_assets/scripts/check_site.py
conda run -n base python -B website_assets/scripts/check_generator.py
```

`check_site.py` checks the seven actual models, offline/file opening, downloads, interactions, and responsive layout. `check_generator.py` performs a real custom Blender build through the browser and checks request validation and caching. Both write reports and screenshots to `preview/`.

Native fidelity checks (requires the `--save-blend` snapshots):

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python website_assets/scripts/check_v2_export.py -- baseline
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python website_assets/scripts/check_v2_export.py -- surface
```

These compare every exported mesh vertex and triangle index against the saved native scene, verify source hashes, and compare browser-style joint transforms to the original add-on's posed transforms. Both models passed: 90 meshes each, 16 joints each, and maximum posed matrix error below `2.4e-7`.

The hero clip was produced from the original `assets/grasp.mp4`:

```powershell
ffmpeg -y -i website_assets/assets/grasp.mp4 -t 9 -an -vf "scale=960:-2,fps=24" -c:v libx264 -crf 24 -preset fast -pix_fmt yuv420p -movflags +faststart website_assets/site/media/hero.mp4
```

Research copy and reported values are based on the existing project page and local IROS presentation. The original PNG/JPEG figures and videos remain the authoritative presentation of experimental results.
