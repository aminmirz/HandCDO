# HandCDO research webpage

Open `website_assets/index.html`, or serve it from the repository root:

```powershell
python -m http.server 8080 --directory website_assets
```

Visit http://localhost:8080. No build step, server backend, CDN, or external font is required.

## v2 page (interactive)

A second, newer page lives at `website_assets/v2/index.html` (http://localhost:8080/v2/). It leaves the page above and all its media untouched and reuses them through `../` paths. It adds:

- A live 3D stage showing the seven `generation_v2` exports in `site/models/` (drag to turn, flex and spread sliders, open/close animation, preset switching, per-design configuration download). It uses the vendored `site/vendor/three.min.js` and never approximates geometry.
- A scroll-driven walkthrough of the co-design loop (sampling, generation, demonstrated task grasp (rendered components only), grasp-pose search, joint-space search, wrench-space score).
- Raw human demonstration videos are intentionally not shown. `scripts/build_v2_media.py` still writes `v2/media/demo-*.mp4`, but no page references them; leave them out when publishing.
- Result cards for the three fabricated designs, cropped from the paper figure.

New media is in `v2/media/` and is rebuilt with:

```powershell
python -B website_assets/scripts/build_v2_media.py
```

`v2/media/manifest.json` records each source file, crop rectangle, and demo speed-up.

### Overview slide, loop clips, and real-world sections (latest)

- **Overview.** This is the IROS presentation's framework slide (slide 3), exported from PowerPoint at 3600 px with its GIFs and title hidden (`v2/media/overview/framework.webp`). The slide's six animations play on top as live clips, at the slide's exact positions and crops (`v2/media/overview/*.mp4`, multiply-blended so the slide's labels stay on top). Hover regions follow the slide geometry, and a spotlight veil dims everything except the active stage.
- **Sampling step (A).** The slide's radar panel is exported without its dots (`v2/media/components/radar-panel.webp`). `app.js` runs a small TPE illustration on top in slide coordinates: 8 new samples per iteration (the config's `n_samples_per_iter`), drawn around the current good set, then re-split into the top 25% (green) and the rest (red).
- **Loop clips.** `scripts/build_framework_clips.py` renders `scripts/render_framework_clips.py` in Blender, then encodes to `v2/media/clips/`:
  - **Hand and grasp (close, wrench).** These clips use the exact hand of the paper's `grasp.mp4` (encoded 2026-03-06; its last frame equals `media/teaser_handgen/graps/asd.png`). That is `media/teaser_handgen/hand/assembled_hand_model.blend`, the hand generated that day: flat palm with no pads. `hand anime.blend` is a later version with pads and a re-keyed animation, and only its hammer is used. `framework_scene.open_anime_exact` opens the hand and appends the hammer at its stored palm-frame pose. The clip's own animation was never saved, so it was recovered from the clip: camera (80 mm perspective) and joint keyframes at clip frames 0, 10, ..., 50, fitted by colour-class matching (`scripts/data/anime_grasp.json`). E plays those keyframes with linear blends over the clip's 51 frames, as the original moves. Each frame is checked against the hammer with a BVH overlap test, and a digit that would penetrate is pulled back toward its previous angles. D samples only the finger abduction joints (`j1`) and the first two thumb joints (`D_RANGE`), with the other joints held open and no new digit-digit or digit-hammer contact; F applies the wrench test to the final grasp. The site uses the front versions (`close_front`, `wrench_front`: orthographic, looking into the palm, rolled so the fingers point up); `close` and `wrench` keep the grasp.mp4 view. In `wrench_front` the glyphs sit on the hammer head, away from the hand (`front_anchor`): in-plane forces push onto the head or pull off its end, forces along the viewing axis are drawn on a screen diagonal (opposite diagonals for + and −), torque rings about in-plane axes are tilted towards the camera so they read as ellipses (the one about the handle wraps it beside the head), and the camera frames the hand, tool and all glyphs. The glove demonstration (`task`) is unchanged.
  - **`task`:** only hammering is shown (`DEMO_SHOWN`), with no prop under it (`PROPS_SHOWN` is empty); stirring and cutting, and their bowl and board, are still in the code. The MANO right hand (mean shape, the mesh HaMeR and WiLoR regress; `scripts/mano_hand.py`, posed with MANO's linear blend skinning and pose correctives in HaMeR's light blue) grasping each tool in turn: hammer, spoon (stir) and knife. Grips are set per tool in `GRIPS` (hammer: handle diagonal, head out of the thumb side; spoon: fist, bowl out of the little-finger side; knife: overhand, blade forward, edge down). The tool is lifted off the open hand until clear, then each finger closes until its mesh touches the handle and its outer joints curl on; the thumb lifts off the palm, then wraps. Hand and tool follow the tool's tracked 6-D trajectory (`grasp_evaluation/data/{hammer,stir,knife}/trajectories`; hammer frames 240-390, the five strikes, at the recording's 30 fps; 3-frame smoothing, which keeps the 25 cm head swing that an 11-frame window shrank by ~19%; framed over the whole segment). Checked against `rgb.mp4`: the poses are in metres in the camera frame, one per video frame, and the projected hammer mesh (30 cm) lies on the real hammer. The hand grips 23 cm from the head, the centre of the demonstrator's grip in the video; the MANO mean hand (wrist to middle knuckle 9.5 cm) is used at true scale, with the tool-tip path and a pose triad drawn; the path is one smooth tube (`fading_trail`: Catmull-Rom through the points, rebuilt each frame) that fades with age (`TRAIL_FADE` = 60 frames: a gradient texture along the tube blends older parts to white, and the radius thins to nothing). Each recording is turned so gravity points down: up is the hammer's swing direction, the normal of the level stirring circle, and the knife's edge-up blade axis. The view is side-on to the hammer handle, from the front and above the bowl, and side-on to the blade. A plywood block (with a nail), a bowl and a cutting board sit at the lowest point the tool reaches, so it touches but never sinks in. The demonstrations record no hand pose, so the grasp placement is illustrative. MANO files are licence-restricted and not committed (`scripts/data/mano/` is git-ignored): put MANO_RIGHT.pkl from mano.is.tue.mpg.de in `scripts/data/mano/models/` and convert it with `python scripts/convert_mano.py <pkl> scripts/data/mano/models/mano_right.npz` (plain numpy: template, faces, joint regressor, weights, posedirs). `glove_hand.py` and `human_hand.py` are earlier stand-in hands, kept but unused.
  - **`close`:** joint-space samples (D), then closing on contact (E).
  - **`wrench`:** the 12 wrench tests (`f±x … t±z`). The tool moves in the grasp under each push or twist.
- **Real-world sections.** Each row is one video per task (`v2/media/experiments/{stir,hammer,cut}.mp4`), cut straight from the talk video `HandCDO IROS Presentation/assets/exp_exported.mp4`. The cells keep the exact crop and pixels of the talk (rows y 0–401, 402–777, 778–1079; columns x 0, 292, 585). In the talk, the hammering and cutting cells were out of hand order; the build re-orders them (`ORDER`) so every row reads high, mid, low.
  - **Timing.** Each section covers exactly the window in which that row's slip plot draws: from 0 s (stir), 2.41 s (hammer) and 2.94 s (cut). These were measured from the plot's drawing tip, which advances at the axis scale, so video time equals plot time.
  - **Build.** `scripts/build_experiment_sections.py`. The older raw-camera re-cut (`scripts/align_experiment_cells.py`) is kept for reference only.
  - **Live plots.** The curves are recomputed from the OptiTrack data with `experiments/stability_analysis.py`, using the post-processing of `stability_analysis_animated.py` (`scripts/export_slip_curves.py` writes `v2/media/experiments/slip.js`). They are drawn live from each video's clock. Hovering a hand highlights its curve.

### Pipeline overview (earlier version)

"The full pipeline" (`v2/pipeline.js`) replaces the earlier three-panel teaser with one diagram:
- **Design optimization:** A sample, B generate.
- **Grasp optimization in simulation:** the demonstrated task grasp, then C → D → E → F, iterating.
- **Build & test:** fabrication, then real-world tool use.

Hovering, focusing, or tapping a stage dims the others, highlights that stage's arrows, and opens a card with a description, settings from the paper's default optimizer config, a clip or image, and a link to the matching section. The stage clips are the IROS presentation's framework animations (`assets/{general,surface,fingers,graspsample,sidejoints,grasp}.mp4`), cropped and re-encoded by `python -B website_assets/scripts/build_v2_media.py --pipeline` into `v2/media/pipeline/`.

### 3D stage designs

The v2 3D stage shows nine designs from `v2/models/` (the seven exports in `site/models/` are kept but no longer shown): compact palms, a side finger, thumbless radial/tri-claw/star layouts, fingers on the top and bottom edges, and a large palm with longer links and four-joint digits. `scripts/creative_hands.py` defines them. Each digit sits at a location t on the closed palm outline (counter-clockwise from the right side: top 0.25, left 0.5, bottom 0.75), with its own code, added link length, and angle. They are built with the native `PalmConfig`/`Hand`, assembled and given joints by the add-on's own functions, and exported with `export_v2_model.export_scene`.

```powershell
python -B website_assets/scripts/build_creative_hands.py --sketch layouts.png   # quick top-view layout check, no Blender
python -B website_assets/scripts/build_creative_hands.py                        # export all designs (about 1 minute)
```

Each folder holds `model.js`, `design.json`, `configuration.zip`, and a Workbench `preview.png` in the approved style.

The hands are built with the generator's **Detailed Viz** palm (`detailed_viz=True`: thumb base mount, screw holes, hand mount, screw attachments, motor and USB cable cut-outs). They are shown upright with the palm facing the camera. Flex and Spread ranges are collision-checked per hand at export (`scripts/motion_limits.py`, BVH mesh overlap in Blender, ignoring contacts already present at rest) and stored in `model.js` under `parameters.motion`:
- **Flex 100%:** each digit's furthest closing fraction before it newly touches the palm.
  A thumb is also stopped before it runs into the fully flexed fingers, with a 0.05 margin. Only Long reach has been re-exported with this rule so far (thumb 0.59 → 0.36); the other hands keep their earlier limits.
- **Spread ±100%:** the furthest side-joint rotation in each direction before two fingers touch. The slider only offers directions with room to move.

### Figure components

Teaser, framework, design-space, results, and experiment visuals on the v2 page are the original component images from `HandCoDesign/figures.pptx`, each cut with the same crop the slide uses (not crops of the flattened figures). Rebuild with:

```powershell
python -B website_assets/scripts/build_v2_components.py   # --figures <path to figures.pptx>
```

`v2/media/components/manifest.json` lists the slide, shape name, embedded image, and crop for every component.

### Interactive SHAP plot and parameter clips

Each of the 28 rows in the paper's SHAP plot is a hover/focus/tap target (`v2/shap.js`, `v2/shap-data.js`). The floating card shows the parameter's meaning, the optimizer's search range (from `iros_paper_codebase/optimization/nested_optimizer_v2.py`), and a clip of a `generation_v2` hand sweeping that range. Hovering a bar in the group chart highlights its rows.

The 28 clips use the approved rendering setup (Workbench, `mylight.sl`, Both cavity 2.5/2.5 and 2/2, original `components.blend` materials, white background, front orthographic camera at scale 0.54):

```powershell
python -B website_assets/scripts/build_shap_parameters.py --preview   # min / mid / max keyframes
python -B website_assets/scripts/build_shap_parameters.py             # all frames, then encode to v2/media/params/
```

Digits follow the paper's base hand: F0 is the right finger beside the thumb (green), F1 the middle (yellow), F2 the left (blue). Finger side offsets use the paper's straight tangent translation with the finger direction kept. The finger and thumb code clips show two `generation_v2` codes as stand-ins for the paper's two code options (`110-11` / `000-03` fingers, `1-2` / `0-2` thumb), since the code grammars differ. Files served by the v2 page: `v2/`, `site/models/*/model.js` and `configuration.zip`, `site/vendor/three.min.js`, `site/media/`, `figs/`, `assets/exp_exported.mp4`, and `summary_video.mp4`.

The white academic page contains the project title and authors, abstract, co-design framework, eight approved parameter animations, simulation results, physical experiments, project video, and a copyable BibTeX citation. Animations are grouped into four keyboard-accessible tabs. Figures can be enlarged. Videos load when visible and pause offscreen; reduced motion disables automatic playback and scroll transitions. Parameter animations have no native hover controls; the separate Play/Pause animations button controls playback, including when reduced motion is enabled. The full project and experimental videos retain native controls.

## Approved media

The page uses `site/media/approved/`, not the earlier renders in `site/media/parameters/`. All original exports are preserved under `preview/`.

| Web clip | Approved original | Duration |
| --- | --- | --- |
| palm-geometry | preview/palm-parameters/videos/palm-parameters-overview-7s.mp4 | 7 s |
| digit-placement | preview/thumb-assembly/thumb-and-fingers-7s.mp4 | 7 s |
| finger-offsets | preview/individual-finger-offsets/individual-finger-offsets-4s.mp4 | 4 s |
| base-angles | preview/combined-base-angles/combined-base-angles-3s.mp4 | 3 s |
| link-lengths | preview/independent-link-lengths/independent-link-lengths-3s.mp4 | 3 s |
| joint-configurations | preview/simultaneous-kinematics/simultaneous-kinematics-5s.mp4 | 5 s |
| contact-surfaces | preview/contact-surfaces-front/contact-surfaces-front-5s.mp4 | 5 s |
| fingertip-geometry | preview/fingertip-geometry/fingertip-geometry-5s.mp4 | 5 s |

Web copies retain the approved front views, original generator colors, white backgrounds, cavity shading, and timings. A fixed crop for each clip covers the union of its native projected geometry bounds, removes the baked title/footer, and pads to 800 x 600. HTML provides captions and controls. `site/media/approved/manifest.json` records source paths, crop rectangles, and poster times. Geometry and materials are not regenerated for this export.

Rebuild web media with FFmpeg and Python:

```powershell
python -B website_assets/scripts/prepare_approved_media.py
```

The animations demonstrate the current `generation_v2` generator. The paper experiments use `iros_paper_codebase`; the page distinguishes these.

## Validation

```powershell
python -B website_assets/scripts/check_site.py
```

Requires Python Playwright and Chrome (override the executable with `HANDCDO_CHROME`). Checks cover all eight video decodes, tabs and keyboard navigation, hidden-panel playback, motion controls, reduced motion, downloads, figure dialogs, citation copying, responsive layouts, direct local opening, and the no-JavaScript fallback. Reports and screenshots are written to `preview/`.

## Files to serve

Serve `index.html`, `site/style.css`, `site/app.js`, `site/media/approved/`, the referenced research WebPs in `site/media/`, `figs/`, `assets/exp_exported.mp4`, and `summary_video.mp4` with their relative paths intact. Rendering scripts, `preview/`, `.runtime/`, and legacy media are not needed by the page.

All changes are confined to `website_assets`. The root `index.html` remains the original website entry point; the redesigned page is `website_assets/index.html`.

## Publishing (GitHub Pages)
The site is served from the `gh-pages` branch (no code there; `main` holds the code). `scripts/build_pages.py` assembles a self-contained copy of `v2/` in `preview/pages/`: the page at the root, the files it uses from `site/`, `figs/` and `summary_video.mp4` copied in (only the copies are rewritten; the Classic page link is dropped), `.nojekyll`, and the sources in `_source/` (MANO files excluded). It fails if a reference points outside the site or at a missing file.

```
py -B website_assets/scripts/build_pages.py
cp -r website_assets/preview/pages/. ../HandCDO-pages/      # git worktree of gh-pages
cd ../HandCDO-pages && git add -A && git commit -m "Update website" && git push
```
The worktree was created with `git worktree add --orphan -b gh-pages ../HandCDO-pages`. Pages source: Settings → Pages → Deploy from a branch → `gh-pages` / root.


## This folder
Sources of the page served from this branch. The page itself is `v2/` in `website_assets/` (published here at the root by `scripts/build_pages.py`); the clips are rendered by the `scripts/build_*.py` / `render_*.py` scripts described above. MANO model files are not included (licence).
