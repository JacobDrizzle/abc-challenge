# tactical/ — Anduril-style rebuild of the story page

Started 16 Nov 2026 as a copy of `story/` (minus archive_1600, notes, spec, incoming, research, tools).
`story/` and `proposal-team/` are not touched by this work.

## 1. What Anduril does (scraped from anduril.com/bolt and /sentry, 16 Nov)

Reference only: patterns, not their code, assets or fonts.

**Stack.** Active Theory's in-house WebGL framework "Hydra" (not three.js, though it looks like it), with
Lenis (smooth scroll), Theatre.js (keyframed camera moves), Draco meshes + KTX2/Basis textures, decoded in
worker threads. The whole scene is described in a config file (`uil.json`), one entry per mesh/hotspot.

**One template for every product.** Bolt, Sentry, Roadrunner, Ghost, EagleEye all use the same
`Product3DScene`: an `OrbitCameraLayer` camera, a `modelGroup`, and N hotspot points.

**Hotspots** (`Product3DPointMesh`): an invisible sphere parented to the model, with a slug
(`2-axis-eo-ir-gimbal`) that links it to a numbered item in the side panel. Each one stores its own camera
pose: `zDistance, xOffset, yOffset, zOffset, rotationX, rotationY, minZoom, maxZoom`. Clicking a dot or the
list item tweens the orbit camera to that pose; prev/next arrows step through them.

**Dots hide behind the model.** Each hotspot draws one invisible pixel with a WebGL2 occlusion query
(`ANY_SAMPLES_PASSED`). Zero samples passed means the model is in front, so the dot fades out.

**Off-centre framing.** The camera has a `viewportFocus` vector: after aiming at the target it slides
sideways so the target lands left of centre, leaving room for the panel on the right. In three.js the same
effect is `camera.setViewOffset(...)`.

**Idle life.** A small camera wobble (`wobbleStrength`, `wobbleSpeed`), so the scene never sits still.

**Look.** Background `#010101`, bone `#f1f0ea`, acid yellow `#dff140` (hotspot labels, focus rings), red
`#ff3535` (rare), greys `#565654 #b0b0a9 #2f3234`. Big tight display type (Helvetica Now Display) and a
mono for the in-scene labels (Elios). Both are paid fonts; we use **Geist + Geist Mono** (OFL, already in
`fonts/`). Thin 1 px rules, numbered rows `01 02 03`, `+`/`—` accordions, round outline buttons.

## 2. The journey (Nic's idea, 16 Nov)

One continuous zoom, "powers of ten" style:

| # | Scene | Hotspots (dot → panel item) | Click goes to |
|---|---|---|---|
| 0 | TAKE CONTROL hero: headset behind giant type | — | Launch Interactive View: type fades, camera pans in |
| 1 | VR head + headset | IMU, display, **processor** | processor → 2 |
| 2 | Inside the headset: circuit board | IMU chip, memory, **SoC** | SoC → 3 |
| 3 | The SoC die | CPU cores, GPU, **NPU / our 44 KB model** | NPU → 4 |
| 4 | The data | IMU samples → quaternion (w,x,y,z) → float32 bits, 400 samples = 4 s | → 5 |
| 5 | The model (existing `viewer.mjs` scene) | Input, Features, Mirror, Projection, Blocks 1–3, Readout, Trajectory, Forecast | Forecast → 6 |
| 6 | Zoom back out through die, board, headset | — | lands on 7 |
| 7 | Head prediction: ghost headset 200 ms ahead, error readout | — | the rest of the (restyled) story |

### How the zoom between scenes works (proposal)

Nesting a chip die inside a head in one scene breaks depth precision (sizes span ~10⁶). Standard trick:
**match-cuts**. Each scene is its own three.js scene sharing one renderer. The camera flies into a
surface until it fills the screen (the processor's lid, the die), then the next scene starts from a framing
whose first frame looks the same, with a short bright/scanline flash to hide the swap. Going back out is
the same in reverse.

### What has to be built

- `hotspots.js`: anchor list per scene, projected dots (DOM, so they stay sharp and clickable), occlusion,
  hover label, click → camera tween, keyboard + screen-reader list.
- `journey.js`: scene registry, match-cut transitions, prev/next, close, URL deep link (`#scene-3`).
- Board + SoC scenes: procedural three.js (boxes, instanced pads, traces as lines). No downloaded models.
  Labelled generically ("XR processor", "NPU"); no real chip brand or logo.
- Data scene: real quaternions from `traces/`, shown as numbers and their float32 bits.
- Model scene: reuse `viewer.mjs` through its existing `K8Viewer` API; its stage labels become hotspots.
- Theme: new `tactical.css` tokens, applied to the whole page (Nic: whole-page restyle).

## 3. Decisions

Decided by Nic, 16 Nov:
1. **Click-driven** journey, like Anduril (hotspots, side panel, prev/next), not scroll-driven.
2. **Chip:** a proper-looking model if we can build or find one (a freely licensed 3D model is allowed);
   otherwise a generic labelled SoC.
3. ~~The journey replaces the middle of the old scroll story (task/data/model decks).~~ **Revised 16 Nov:**
   the journey replaces **only the model deck**. Page order: hero → challenge → data (the story's own
   scroll section, kept) → **TAKE CONTROL + Launch Interactive View** (where "Now drive it yourself" sits
   now) → results → team → paper. A one-line summary and the picture fallback stay under TAKE CONTROL
   for readers who don't launch it.
4. **No data level in the journey.** A live bit-wall level (`proto/datalevel.js`) was tried between the
   die and the model and dropped: unclear, it shows bits without explaining the data. The journey is
   head → board → die → model.

Still open:
5. Model level: A (etched die), B (console), or both (B as an expert view inside A). Switch in the proto.
6. Phone: full journey, or a reduced version?

## 4. Suggested build order

1. Theme tokens + TAKE CONTROL hero + launch transition on the existing head (no new scenes yet).
2. Hotspot system on the head (scene 1), with the side panel. This is the Anduril look end to end.
3. Model scene hotspots (5), since the scene already exists.
4. Board + SoC scenes (2, 3) and the match-cuts.
5. Data scene (4) and the zoom-out ending (6, 7).

## 5. Status (16 Nov 2026, evening)

Built and live in `tactical/index.html`: story hero → data scroll (colours only changed) → **TAKE CONTROL** (`tc.js`,
`tactical.css`; journey in `journey/` via iframe, head turns to face the reader on scroll-in, furniture slides out on Launch)
→ results → team → paper. Journey: head → board → die → model (A etched / B console / 3D viewer switch) → **SEE IT PREDICT**
climb-out → prediction view (ghost 200 ms ahead, live error vs head-held-still). Anduril-style side panel on the 3D levels.
Fixes: Launch button pointer-events (`all: unset`), story.js band/carry softened for the TAKE CONTROL gap.
`proto/` is the sandbox; `archive_pre_tc/index.html` is the page before the swap.

Open: phone layout + real-phone test · A vs B (team) · ghost tint · realistic board/chip models · GitHub Pages paths
(`../adapter.js`, `../vendor`) · tidy `proto/` vs `journey/` duplication.

Headset model (16 Nov, late): `models/headset.blend`, collection HEADSET = v2, a generic "Quest-class" standalone design
(pillowy matte one-piece visor ≈182×72×99 mm at page scale 1 unit ≈ 275 mm, 4 corner cameras, grey face cushion, short rigid
side arms, split soft rear strap into a rear plate, top strap). No logo, no Quest-3 three-pill sensors (public site). v1 halo
version kept hidden in HEADSET_v1. Previews in `models/previews/` (q5_* = v2). Not yet exported: procedural fabric needs baking,
then glTF + GLTFLoader + hero.js swap (tactical only).

## 6. Board level, real-board version (16 Nov 2026, night)

Research: `research/quest3_board.md` (iFixit Quest 3 chip ID). ~~Decided (Nic): chips have neutral tops, no logos.~~
**Revised (Nic, 16 Nov, late): replicate the real board exactly, chips carry their real markings** (Qualcomm SXR2230P, SK hynix,
WCN6856, Qorvo QM45392/QM42391…). Meta's ∞ logo on the silkscreen is left out (same as the headset); board number kept.

Build (v1): `models/board_build.py`, run headless (`Blender -b -P`), writes `models/board.blend`, `assets/board_q3.glb` and
previews in `models/previews/board_*`. Side 1 only (the SoC side), laid out like the real board's photo:
- outline: wide body, raised centre tab, two lower lobes, notch at the bottom middle (where the fan sits); 1 page unit = 10 mm,
  board ≈ 100 mm wide (scale from the photo's proportions; absolute size is an estimate)
- centre shield frame around memory (left) + SoC (right); right shield frame around Wi-Fi SoC, two Wi-Fi front ends, Bluetooth
- audio DAC, IR front end, ~10 board-to-board flex connectors along the edges, two tall capacitor cans, mounting holes, QR label
- hundreds of 0201/0402 passives with metal end caps, vias, faint surface traces under black solder mask
- named empties for hotspots: `anchor_soc`, `anchor_mem`, `anchor_wifi`

Open for Nic: the board's "Motion sensor" hotspot. The headset IMU is not on the main board's public chip list, so the board
level can't point at a real IMU part. Options: drop it, or swap it for Wi-Fi / the STM32 (side 2, would need the flip side).

v2 (replica): `models/board_build.py` now measures every part off iFixit's step-1 photo (≈0.0615 mm/px, board ≈100 mm):
outline, both shield frames (lids off), memory + SoC + Wi-Fi block with real markings, 10 B2B connectors with silk brackets,
3 U.FL sockets (ANT0/ANT1), 3 wire headers, 4 steel standoffs, plated holes, 2 cans "100 VFT A73", status LED, serial label,
IR front end in its dashed box, ~270 passives in the photo's groups. The generic v1 is kept as `board_build_v1_generic.py`.
Previews: `models/previews/board_top.png` (straight down, to compare with the photo), `board_wide.png`, `board_soc.png`.
Not in the journey yet. glb 2.0 MB (compression still to do).

In the journey (16 Nov, night): `journey/journey.js` buildBoard loads `assets/board_q3.glb` (placeholder stays as fallback),
hotspots Wi-Fi 6E / Memory / Processor (XR2 Gen 2) on the replica's anchors with real part names (the board's motion-sensor
hotspot was replaced by Wi-Fi: no IMU on this board's public chip list). Product-shot lighting + studio reflection map
(`boardEnvironment`). Connectors rebuilt as open sockets (housing, cavity, centre tongue, gold contacts). Code-drawn textures in
`journey/board_textures.js`: copper traces/pours/vias under the mask, dimpled tin on the shield frames, grain on chips and plastic.
Index credits gained a "Circuit board" line (iFixit, trademarks).

Die level (16 Nov, night): redrawn in the style of Apple's die renders (`journey/die_texture.js`: line-art macros, mirrored
units, SRAM hatching, pastel wash; `?die=tactical` = graphite/bone). XR2 Gen 2 block list, layout illustrative. Animated
10 ms data path (`dataPath()` in journey.js): pads → system cache → k8 NPU tile → GPU → displays, eased legs, comet tail,
stop glows, #flow caption ("shown 650× slower", computed from the route). Nic: "feels right".

Fixes (16 Nov, late, from Nic's Safari pass): (1-2, 4) level-change freezes: the match-cut's frozen frame is now a GPU render
target drawn as a fading quad (no drawImage readback); into DOM levels the WebGL canvas itself fades; all 3D levels are drawn once
off-screen at load (texture upload + render-target shader variants). Freeze step 237 ms → 0.4 ms (Chromium). (3) The side panel's
tap magnifier replaced by the etched view's reach tree (live). A/B switch only on the model level; 3D viewer button hidden
(`?model=3d` still works).

Etched view pulse (16 Nov, night): `modelPulse()` in journey/modelviews.js. The die level's light carries on through k8: input →
features → projection → blocks 1-3 → readout → trajectory → forecast, eased legs, comet tail, each stage's box glows. Two lights
(observed lane + mirrored lane, dimmer) through the shared-weight blocks, merging at the readout. Pauses with the clip (space).
Replaces the old looping lane dashes. Note line: "LIGHT: ONE SAMPLE THROUGH K8, SLOWED DOWN".

Data side, funnel layout (16 Nov, night, Nic: option B): `dataview.js` FUNNEL (default; `?dv=classic` = old layout). Funnel bar on top
(recordings left after each step, square-root bars, current acid), per stage one big number (+ "−N at this step"), the headline, one
"what we do" line, one chart; head only on stages 1, 5, 9; numbers table, notes, other views, caption, rule and sources in one fold.
Palette: acid = head/data, bone = kept, greys = dropped (was amber/green/red). Explore section above the team (`#s-explore3d`) and the
3D spaces themed (tactical.css 4-5); the Quest headset in both 3D spaces (head3d.js).

Late 16 Nov (after the Safari pass):
- Level changes fade through black (journey.js diveInto / backOut, fadeTo, #fade): zoom in while fading out, swap in the dark (~6 ms),
  fade in while the new camera settles. The GPU frozen-frame machinery is gone. SEE IT PREDICT is a sideways push (finale): the model
  view slides out left, the head (already in its prediction pose) slides in from the right.
- Console view (B): a scrub bar over the test clip (3 s), dragging pauses; level() gained seek / setPlaying / clip.
- Files: both glbs Draco-compressed (vendor_gltf/DRACOLoader.js + draco/ decoder from three r180; hero_model.js makeGLTFLoader).
  board 2.5 MB -> 270 KB (both sides), headset 3.7 MB -> 533 KB (models/headset_export.py, full mesh, JPEG textures).
  headset.blend stores its textures by absolute path from before the move (nic_scratch): the export script re-points them; opening the
  .blend in the GUI shows them missing until File > External Data > Find Missing Files (models/quest_tex/).
- Board side 2 (board_build.py SIDE 2, from iFixit step 3): 3 shield frames over the Qualcomm PMICs and clock, SK hynix 128 GB storage,
  H330 inductor, STMicroelectronics STM32G071, test pads, passives, label. Journey: Side 1 / Side 2 switch on the board level turns the
  board over (flip group, half a turn about z) and swaps hotspots (STM32, storage, power).

Main site + team copy (16 Nov, late): story/ and proposal-team/ moved to k8_3d/archived_sites/. The team repo gets a copy in
abc-challenge/visualization/ (adapter.js, vendor/, tactical/ minus the Blender working files). Hero poster re-rendered with the Quest 3
(models/poster_render.py). Identifiers removed (tools/scrub_ids.py): BOXRR player/recording UUIDs -> player-0001 / rec-0001 labels
(same label in every file, so lookups still match), machine paths -> plain words; checksums kept. Earlier commits in Nic's repo still
hold the old IDs (git history); the team copy was never committed with them.
