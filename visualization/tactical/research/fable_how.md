# How to build the tactical journey

Design report, 16 Nov 2026. Read-only study of `tactical/`; nothing was changed. three.js here is r185
(`../vendor/three`) with only OrbitControls, CSS2DRenderer and RoundedBoxGeometry vendored: no
post-processing and no glTF loader yet.

Folded in from PLAN.md section 3: the journey is **click-driven**, a **freely licensed chip model is
allowed**, and the journey **replaces the task and model decks**.

## 1. Scene architecture: one renderer, one scene per level, match-cuts

| Option | Verdict |
|---|---|
| **A. One renderer, a `THREE.Scene` per level, match-cut** | **Recommended.** Each level has its own units and near/far, so depth precision is never a problem, and one scene is drawn per frame. |
| B. One nested scene, `logarithmicDepthBuffer`, floating origin | No. Sizes span about 10⁶, log depth is slow on phones, and nothing hides the 188-triangle headset shell up close. |
| C. Live render-target portals | Not in v1: two scene renders per frame on every approach. Later extra: a still of the next level as a texture on the chip lid. |

Hiding the swap, a "freeze-frame dissolve":

```js
async function dive(next) {
  await flyTo(cur.exitPose, easeInExpo);            // a dark surface fills the frame (visor, lid, NPU block)
  freeze.getContext('2d').drawImage(renderer.domElement, 0, 0);   // same frame as render()
  await renderer.compileAsync(next.scene, camera);  // no shader-compile stall on the first frame
  cur = next; setPose(next.entryFar); flyTo(next.entryNear, easeOutExpo);
  freeze.animate([{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(1.6)' }], 350);
}
```

The old frame keeps growing as it fades and the new scene starts slightly far and settles, so both move
forward. Zooming out uses scale 0.6. The overlay is CSS, so it costs no WebGL pass. Interpolate distance
in log space (`d0 * (d1/d0) ** e`) so the zoom speed feels constant.

## 2. Hotspots

- **Data:** `{ id, anchor, normal, pose: { target, az, pol, dist }, focus }`. The anchor is an empty
  `Object3D` added to the mesh (`hero.headset.add(a)`), so it follows the head.
- **Dots are DOM `<button>`s.** Each frame: `a.getWorldPosition(v).project(camera)`, then
  `transform: translate(x, y)`. `project()` includes the view offset. Hide when `v.z > 1`.
- **Occlusion: raycast, 10 times a second.** First a free test: the anchor's normal facing away means
  hidden. Then `raycaster.far = dist * 0.99; intersectObjects(level.occluders)`, using the headset parts
  and a hidden sphere for the head (not the 7,200-triangle skinned mesh). The flat board and die need
  the normal test only.
  - Occlusion query (Anduril's way): `WebGLRenderer` has no helper, so it means raw
    `gl.beginQuery(gl.ANY_SAMPLES_PASSED, q)` in `onBeforeRender`, with results a frame late. Not worth
    it for 3 to 10 dots.
  - Depth readback needs a blocking `readPixels`. Worst option.
- **Camera tween:** lerp the target, slerp the orbit direction (`setFromUnitVectors`, then
  `slerpQuaternions`), lerp the distance. That is `fly()`/`stepTween()` in `viewer.mjs` (line 1136), so
  copy its rule: cubic in-out, cubic-out if interrupted, 800 to 2,000 ms by distance. A straight position
  lerp would cut through the head.
- **Off-centre framing:** `camera.setViewOffset(W, H, +panelW / 2, 0, W, H)` moves the subject left;
  tween it as the panel slides in. On phones shift up, above a bottom sheet.
- **Idle life:** rock the model group a degree or two; anchors follow for free.
- **Keyboard and screen readers:** the numbered list in the panel is the real control (buttons with
  `aria-expanded`); dots mirror it. Arrows and prev/next step, Esc goes up a level, an `aria-live` line
  announces "Processor, 3 of 3". Reduced motion: flights become short fades. Touch: labels always shown.

## 3. The model stage: keep `viewer.mjs` on its own canvas

`viewer.mjs` creates its own renderer, scene, camera, loop and labels at module top level and reads about
40 element ids from `index.html`. It **cannot draw into a shared renderer** without a rewrite. Its embed
mode already covers us:

- Start: `K8_NOSTART = null; K8_EMBED = { el: host, user: 'test50', view: 'input', clear: true }`, then
  insert `<script type="module" src="viewer.mjs?embed=j1">` (what `model_panel.js` `start()` does).
- Drive: `K8Viewer.ready`, `showStep(key)`, `setActive(false)` (loop off), `setPlaying`, `enter()`.
- Frame beside the panel: `K8_EMBED.free = { left, top, width, height }`, then `K8Viewer.reframe()`.
- `clear: true` makes its canvas transparent, so the journey canvas below holds one still of the NPU
  while the model fades in above with CSS opacity. Two WebGL contexts in total.
- **Hotspots with no viewer edit:** its stage pills are DOM (`#labels .lbl.stage[data-key]`: input, feat,
  proj, mirror, blk0 to blk2, head, traj, out, gaze). Embedded they are `inert`, so the journey reads
  their positions, hides them with CSS, draws its own dots and calls `showStep(key)`. A 6-line
  `K8Viewer.anchors()` would be cleaner, but that is an edit: Nic's call.
- Follow the viewer's order (`proj` before `mirror`); PLAN lists them the other way round.

## 4. Board, SoC die, NPU

**Procedural** (recommended for die and NPU, default for the board):

- Board: `RoundedBoxGeometry` slab; traces from a seeded router merged into one `BufferGeometry` of flat
  quads; vias and passives as two `InstancedMesh`es (`setColorAt`); silkscreen from a `CanvasTexture`.
  About 20k triangles, 12 draw calls.
- Die: two triangles and a fragment shader. A treemap splits it into blocks (CPU, GPU, NPU, memory, I/O
  ring); a hash grid draws memory cells or logic noise, faded with `fwidth` to stop shimmer.
  `MeshPhysicalMaterial({ metalness: 1, roughness: .25, iridescence: 1 })` gives the die-shot sheen.
- NPU: instanced tiles, our 44 KB as one small lit patch, captioned "illustrative, not to scale".
- Light: one grazing key light plus the `PMREMGenerator` room `stage_task.js` builds (line 207). No
  shadows, no `EffectComposer`: CSS vignette and scanlines, additive halos. About 15 KB of JS per scene.

**Free models** (licences read from the Sketchfab API and Poly Pizza today; not viewed or downloaded):

| Model | Licence | Size |
|---|---|---|
| "Circuit Board", Daedron (Sketchfab `7d667eb9648d48bb9a91cd7d2ca456c8`) | CC-BY 4.0 | 9,478 faces, 10.5 MB, 4 textures |
| "Drone Control PCB v3", Plasmabot (`839dfadc63cf42a3a50e7bfb04f3979e`) | CC-BY 4.0 | 89,138 faces, 1.2 MB, no textures |
| "Unmarked processor + Socket", 0004707 (`26d2147e082648949b9c703c514e6901`) | CC-BY 4.0 | 189,142 faces, 17.7 MB; author says only the FBX is complete |
| "Microchip", iPoly3D (poly.pizza/m/tIp60lIg43) | CC0 | low-poly, size not shown |

Ruled out: branded models (Ryzen, Raspberry Pi, RTX 4090), Sketchfab "Free Standard" (not Creative
Commons), SA or NC licences, KiCad's library (CC-BY-SA). No free die with an NPU exists, so the die is
procedural either way. A glTF needs `GLTFLoader.js` and `BufferGeometryUtils.js` (r185) vendored,
`gltf-transform optimize` to under 1.5 MB, and a credit in `#s-credits`.

## 5. Data scene

`traces/test50.js` (3.7 MB): `initial_context` (400 × `[w,x,y,z]`), 1,092 `steps` each with
`input_quaternion`, `features`, block activations, and for the 300 scored steps `prediction`,
`ground_truth` and both errors. `hero/clip.js` (110 KB): 1,200 samples of `q`, `pred`, `err`, `hold`, same
player. **Use the clip**: it is already loaded for the head.

```js
const f = new Float32Array(1), u = new Uint32Array(f.buffer);
f[0] = 0.9994549;  u[0].toString(2).padStart(32, '0');   // "00111111011111111101110001000111"
```

- The newest sample as a mono line splits into 4 × 32 bit cells, coloured sign, exponent, mantissa.
- "Bit wall": one quad with a 400 × 128 `DataTexture` (`RedFormat`, `NearestFilter`), one column
  rewritten every 10 ms. Two triangles, 51 KB.
- The wall squashes into the 4 × 400 strip and cuts to the viewer's `input` step, using the existing
  `K8Viewer.arrive(1 → 0)` that slides the input strip into place.
- Honesty: the clip rounds to 6 decimals (the trace to 7), so the last bits can differ from what the
  model saw. The trace records one new quaternion per call with history in caches; check the "400
  samples in" wording with the team.

## 6. Performance, phones, fallbacks

- One loop, drawing only the active level and only on change; stop on `document.hidden`.
- `setPixelRatio(Math.min(devicePixelRatio, 1.5))`; antialias on the full tier only.
- Per level: desktop 60k triangles and 40 draw calls; phone 30k and 25, `MeshStandardMaterial`, raycast
  at 5 Hz. Reuse `K8_BOOT.tier`, `.phone`, `.live3d`.
- Never more than two contexts. Build the next level in `requestIdleCallback`; start the viewer while
  the reader is on the SoC, so the data scene covers its load.
- No WebGL or context lost: `hero/poster.webp`, and the panel becomes a numbered article with
  `pics/step_*.webp` plus new stills taken at each hotspot pose. `lite.js` is unchanged.
- GitHub Pages: the import map and `adapter.js` point to `../`, outside `tactical/`. The boot script
  relabels parser-added module scripts, so inject `journey.js` after `DOMContentLoaded`.

## 7. What happens to the old files

- **`story.js`**: no code change. It drives whatever `.dk-track` exists, so it keeps running the results
  deck; call `K8Story.suspend()` while the journey is open. Keep the task and model steps' HTML as the
  plain-article fallback and fill the accordion from those elements, so the words exist once. Old links
  (`#s-model`, `#s-explore`) need redirects.
- **`stage_task.js`**: retired here. `journey.js` owns the renderer, draws the hero in place and goes
  fullscreen on launch (same canvas). Reuse `hero.js` unchanged (`createHero`: `group`, `headset`,
  `ghost`, `update(t)`), plus its loader list, lights and readout rule (`err[i − 20]`). `scene_task.js`'s
  scroll camera path is unused.
- **`model_panel.js`**: retired. Port `start()`, `onReady()`, the state mapping, context-loss handling and
  `enterViewer()` into a small `journey_model.js`; drop `grow()`, `edge()` and the debounce.
- Delete nothing until the last phase. Open: PLAN does not say what the **results deck** becomes.

## 8. Build order

| # | Demo at the end | Days |
|---|---|---|
| 0 | Spikes (below) | 1 |
| 1 | Theme, TAKE CONTROL hero, launch move on the existing head | 1 |
| 2 | `hotspots.js` and the panel on the head | 2 |
| 3 | Model stage with hotspots | 1 |
| 4 | Board, die, NPU and the dives | 3 |
| 5 | Data scene and the cut into the viewer | 1.5 |
| 6 | Zoom back out, prediction scene (ghost and readout exist in `hero.js`) | 1 |
| 7 | Cut-over: hide old decks, redirects, stills, phone sheet | 1.5 |

Prototype first:

1. **The dive into a 188-triangle headset**: does the visor-fills-frame cut look right, and does
   `drawImage(webglCanvas)` work on iOS Safari?
2. **Two contexts on a phone**: journey canvas, embedded viewer and the 3.7 MB trace together.
3. **The viewer embedded outside a deck**: label positions readable, `free`/`reframe` framing right.
4. **The die shader's look**: taste work that can eat days. Time-box it; a model is the fallback.
